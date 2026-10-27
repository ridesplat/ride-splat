"""Score held-out renders against clean targets: PSNR, SSIM and LPIPS on the full frame.

    python evaluate.py --pred output/rs2/renders --gt data/ride-splat-dataset/rs2/eval --out metrics.json

Renders and targets are paired by file name. Per frame: PSNR = 10 log10(1 / MSE); SSIM with an
11-tap Gaussian window (sigma 1.5) on RGB in [0, 1]; LPIPS with the VGG backbone. Reported values are
means over frames. No color or exposure correction is applied.
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def read_rgb(path):
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def stem(filename):
    """File name without its image extension; 'x.png.png' (a 3DGS image_name plus .png) -> 'x'."""
    name = filename
    while name.lower().endswith(IMAGE_EXTS):
        name = os.path.splitext(name)[0]
    return name


def images_by_stem(folder):
    return {stem(f): os.path.join(folder, f)
            for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)}


def psnr(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    return 99.0 if mse <= 1e-12 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim(pred, gt):
    from skimage.metrics import structural_similarity

    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


class Lpips:
    def __init__(self, device):
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net="vgg").to(device).eval()

    def __call__(self, pred, gt):
        t = lambda a: self.torch.from_numpy(a).permute(2, 0, 1)[None].float().to(self.device) * 2.0 - 1.0
        with self.torch.no_grad():
            return float(self.model(t(pred), t(gt)).item())


def evaluate(pred_dir, gt_dir, use_lpips=True, device=None):
    preds, gts = images_by_stem(pred_dir), images_by_stem(gt_dir)
    stems = sorted(set(preds) & set(gts))
    if not stems:
        sys.exit(f"No file names in common between {pred_dir} and {gt_dir}.")
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"[evaluate] warning: {len(missing)} target(s) have no render, e.g. {missing[:3]}")
    scorer = None
    if use_lpips:
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = Lpips(device)
    frames = []
    for stem in stems:
        pred, gt = read_rgb(preds[stem]), read_rgb(gts[stem])
        if pred.shape != gt.shape:
            sys.exit(f"{stem}: render is {pred.shape[1]}x{pred.shape[0]} but the target is "
                     f"{gt.shape[1]}x{gt.shape[0]}. Render at the target's resolution (-r 1).")
        row = {"frame": stem, "PSNR": psnr(pred, gt), "SSIM": ssim(pred, gt)}
        if scorer is not None:
            row["LPIPS"] = scorer(pred, gt)
        frames.append(row)
    summary = {"n": len(frames)}
    for key in ("PSNR", "SSIM", "LPIPS"):
        if key in frames[0]:
            summary[key] = float(np.mean([row[key] for row in frames]))
    return {"summary": summary, "per_frame": frames}


def main(argv=None):
    ap = ArgumentParser(description="Score held-out renders: PSNR, SSIM, LPIPS (VGG)")
    ap.add_argument("--pred", required=True, help="folder of rendered held-out views")
    ap.add_argument("--gt", required=True, help="folder of clean targets with the same file names")
    ap.add_argument("--out", default=None, help="write the per-frame and mean scores to this JSON file")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    result = evaluate(args.pred, args.gt, use_lpips=not args.no_lpips, device=args.device)
    s = result["summary"]
    print(f"[evaluate] n={s['n']}  PSNR {s['PSNR']:.2f}  SSIM {s['SSIM']:.4f}"
          + (f"  LPIPS {s['LPIPS']:.4f}" if "LPIPS" in s else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
