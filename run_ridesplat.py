"""RIDE-Splat end to end on one scene: pass 1, self-localization, pass 2, held-out renders, scores.

    python run_ridesplat.py -s data/ride-splat-dataset/rs2 -m output/rs2 --gt data/ride-splat-dataset/rs2/eval

Writes to -m:
    pass1/          joint fit of the scene and R (pass 1)
    weight.png      self-localization weight W from pass 1's R
    pass2/          refit from scratch with W fixed (pass 2)
    renders/        held-out views rendered from pass 2 (scene only)
    metrics.json    PSNR / SSIM / LPIPS against --gt, if given

Each step is skipped when its output exists, so rerunning the same command resumes an interrupted run.
Arguments this script does not recognize (e.g. --refl_l1 0.002) are passed to both training passes.
"""
import json
import os
import subprocess
import sys
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))

# Pass-2 terms used for every number reported in the paper. --minimal drops them.
PAPER_PASS2 = ["--atten", "--atten_max", "0.9", "--appearance", "affine", "--appearance_reg", "10"]


def parse(argv=None):
    ap = ArgumentParser(description="RIDE-Splat: two-pass reflection removal on one scene")
    ap.add_argument("-s", "--source_path", required=True, help="COLMAP scene with train_* and eval_* images")
    ap.add_argument("-m", "--model_path", required=True, help="output folder")
    ap.add_argument("-r", "--resolution", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=8.0, help="smoothing of the self-localization weight, in pixels")
    ap.add_argument("--minimal", action="store_true",
                    help="pass 2 with the weight only, without attenuation or per-frame appearance")
    ap.add_argument("--gt", default=None, help="folder of clean held-out images to score against")
    ap.add_argument("--smoke_test", action="store_true", help="300-iteration passes, to check the setup")
    return ap.parse_known_args(argv)


def pass_commands(args, extra):
    """The two train.py command lines, so they can be inspected and tested without running them."""
    base = [sys.executable, os.path.join(HERE, "train.py"), "-s", args.source_path,
            "-r", str(args.resolution), "--iterations", str(args.iterations), "--eval", "--seed", str(args.seed)]
    if args.smoke_test:
        base.append("--smoke_test")
    out = args.model_path
    pass1 = base + ["-m", os.path.join(out, "pass1")] + extra
    pass2 = base + ["-m", os.path.join(out, "pass2"), "--refl_weight", os.path.join(out, "weight.png"),
                    "--render_dir", os.path.join(out, "renders")] + ([] if args.minimal else PAPER_PASS2) + extra
    return pass1, pass2


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None):
    args, extra = parse(argv)
    out = args.model_path
    os.makedirs(out, exist_ok=True)
    pass1, pass2 = pass_commands(args, extra)
    config = {"source_path": args.source_path, "pass2": "minimal" if args.minimal else "paper",
              "pass1_cmd": pass1, "pass2_cmd": pass2}
    with open(os.path.join(out, "ridesplat_run.json"), "w") as f:
        json.dump(config, f, indent=2)

    refl = os.path.join(out, "pass1", "refl_layer.pt")
    if not os.path.exists(refl):
        run(pass1)

    weight = os.path.join(out, "weight.png")
    if not os.path.exists(weight):
        import torch

        from ridesplat.selfloc import save_weight, weight_from_reflection
        W = weight_from_reflection(torch.load(refl, map_location="cpu"), args.sigma)
        save_weight(W, weight)
        print(f"[self-localization] {weight}: mean weight {W.mean():.3f}")

    renders = os.path.join(out, "renders")
    rendered = os.path.isdir(renders) and any(f.endswith(".png") for f in os.listdir(renders))
    if not (os.path.exists(os.path.join(out, "pass2", "refl_layer.pt")) and rendered):
        run(pass2)

    if args.gt:
        run([sys.executable, os.path.join(HERE, "evaluate.py"), "--pred", os.path.join(out, "renders"),
             "--gt", args.gt, "--out", os.path.join(out, "metrics.json")])


if __name__ == "__main__":
    main()
