#
# Adapted from train.py of 3D Gaussian Splatting, Copyright (C) 2023, Inria, GRAPHDECO research group
# (https://team.inria.fr/graphdeco). Free for non-commercial, research and evaluation use under the
# terms of LICENSE.md.
#
"""One RIDE-Splat training pass: a 3DGS scene fit jointly with a shared, signed reflection image.

run_ridesplat.py chains two passes with self-localization in between. Run this directly for a
single pass or an ablation:

    python train.py -s <scene> -m <output> -r 1 --eval                        # pass 1
    python train.py -s <scene> -m <output> -r 1 --eval --refl_weight W.png \\
        --atten --appearance affine --render_dir <output>/renders             # pass 2 (paper)
"""
import os
import sys
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torchvision
from tqdm import tqdm

from ridesplat.backbone import add_gs_to_path, install_prefix_split, seed_everything

add_gs_to_path()
install_prefix_split()

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import get_expon_lr_func, safe_state  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402

from ridesplat.reflection import ReflectionModel  # noqa: E402
from ridesplat.selfloc import load_weight  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


def build_reflection(args, size, frame_names):
    weight = load_weight(args.refl_weight, size) if args.refl_weight else None
    if weight is not None:
        print(f"[reflection] weight {args.refl_weight}: mean {float(weight.mean()):.3f}")
    return ReflectionModel(size[0], size[1], frame_names, lr=args.refl_lr, l1=args.refl_l1, weight=weight,
                           atten=args.atten, atten_max=args.atten_max, atten_l1=args.atten_l1,
                           appearance=args.appearance, appearance_lr=args.appearance_lr,
                           appearance_reg=args.appearance_reg)


def training(dataset, opt, pipe, args):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("--optimizer_type sparse_adam needs the accelerated rasterizer; see the 3DGS README.")

    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    reflection_state = None
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)
        ckpt_dir, ckpt_name = os.path.split(args.start_checkpoint)
        refl_ckpt = os.path.join(ckpt_dir, "refl_" + ckpt_name)
        if os.path.exists(refl_ckpt):
            reflection_state = torch.load(refl_ckpt)
        elif first_iter > args.refl_warmup:
            sys.exit(f"{refl_ckpt} is missing. Resuming past the warm-up without it would restart R "
                     "from zero mid-training.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_cams = scene.getTrainCameras()
    viewpoint_stack = train_cams.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    reflection = None  # built on the first frame, once the image size is known
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # RIDE-Splat: after the warm-up, fit the observation with the scene plus the reflection model
        if reflection is None:
            reflection = build_reflection(args, image.shape[-2:], [cam.image_name for cam in train_cams])
            if reflection_state is not None:
                reflection.load_state_dict(reflection_state)
        reflection_active = iteration > args.refl_warmup
        if reflection_active:
            pred, appearance_reg = reflection.predict(image, viewpoint_cam.image_name)
        else:
            pred = image

        # Loss. Plain SSIM rather than fused_ssim, as in the runs reported in the paper.
        Ll1 = l1_loss(pred, gt_image)
        ssim_value = ssim(pred, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if reflection_active:
            loss = loss + reflection.penalty(appearance_reg)

        # Depth regularization
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            loss += depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "R": "on" if reflection_active else "warm-up"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in args.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if reflection_active:
                    reflection.step()
                else:
                    reflection.zero_grad()

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save(reflection.state_dict(), scene.model_path + "/refl_chkpnt" + str(iteration) + ".pth")

    reflection.save(scene.model_path)
    return scene, gaussians, background


def render_held_out(scene, gaussians, pipe, background, dataset, out_dir):
    """Render the de-reflected scene (T only) at every held-out camera, named like its target."""
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    with torch.no_grad():
        for cam in tqdm(scene.getTestCameras(), desc="Rendering held-out views"):
            if cam.image_name in seen:
                continue
            seen.add(cam.image_name)
            image = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp,
                           separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            torchvision.utils.save_image(torch.clamp(image, 0.0, 1.0), os.path.join(out_dir, f"{cam.image_name}.png"))
    print(f"[render] wrote {len(seen)} held-out views to {out_dir}")
    if not seen:
        sys.exit("No held-out views were rendered. Train with --eval on a scene that has eval_* frames.")


def prepare_output(args):
    os.makedirs(args.model_path, exist_ok=True)
    print("Output folder: {}".format(args.model_path))
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="One RIDE-Splat training pass")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_dir", type=str, default=None,
                        help="After training, render the held-out views (T only) into this folder.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="300 iterations with a 100-iteration warm-up, to check the pipeline end to end.")
    rs = parser.add_argument_group("RIDE-Splat")
    rs.add_argument("--refl_lr", type=float, default=0.01, help="Adam learning rate for R (eta_R).")
    rs.add_argument("--refl_l1", type=float, default=0.01, help="Weight of the L1 penalty on R (beta).")
    rs.add_argument("--refl_warmup", type=int, default=2000, help="Plain 3DGS iterations before R is used (t_0).")
    rs.add_argument("--refl_weight", type=str, default=None,
                    help="Self-localization weight W (PNG from run_ridesplat.py). Pass 2 only.")
    rs.add_argument("--atten", action="store_true", help="Also fit a per-pixel attenuation A.")
    rs.add_argument("--atten_max", type=float, default=0.9, help="Upper bound on A.")
    rs.add_argument("--atten_l1", type=float, default=0.01, help="Weight of the L1 penalty on A.")
    rs.add_argument("--appearance", choices=["none", "affine"], default="none",
                    help="Per-frame RGB gain and bias on the scene render during training.")
    rs.add_argument("--appearance_lr", type=float, default=1e-3)
    rs.add_argument("--appearance_reg", type=float, default=10.0,
                    help="Weight pulling each frame's gain and bias toward identity.")
    args = parser.parse_args(sys.argv[1:])
    if args.smoke_test:
        args.iterations = min(args.iterations, 300)
        args.refl_warmup = min(args.refl_warmup, args.iterations // 3)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    scene, gaussians, background = training(dataset, opt, pipe, args)
    if args.render_dir:
        render_held_out(scene, gaussians, pipe, background, dataset, args.render_dir)

    print("\nTraining complete.")
