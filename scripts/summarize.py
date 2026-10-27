"""Collect metrics.json from each scene's output folder into one table, next to the paper's numbers.

    python scripts/summarize.py output
"""
import json
import os
import sys

# RIDE-Splat rows of the paper's per-scene tables (PSNR, SSIM, LPIPS), keyed by dataset folder.
PAPER = {"rs1": ("Leesburg", 16.03, 0.6070, 0.5712),
         "rs2": ("Martin", 20.18, 0.8420, 0.4025),
         "rs3": ("Reston", 19.21, 0.7794, 0.4880)}


def main(out_root):
    rows = []
    for scene in sorted(os.listdir(out_root)):
        path = os.path.join(out_root, scene, "metrics.json")
        if os.path.isfile(path):
            rows.append((scene, json.load(open(path))["summary"]))
    if not rows:
        sys.exit(f"No */metrics.json under {out_root}.")
    print(f"{'scene':<10} {'n':>3} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}   paper: PSNR / SSIM / LPIPS")
    for scene, s in rows:
        ref = PAPER.get(scene)
        ref_txt = f"{ref[0]:<9} {ref[1]:.2f} / {ref[2]:.4f} / {ref[3]:.4f}" if ref else ""
        print(f"{scene:<10} {s['n']:>3} {s['PSNR']:>7.2f} {s['SSIM']:>7.4f} {s.get('LPIPS', float('nan')):>7.4f}   {ref_txt}")
    mean = lambda k: sum(s.get(k, float("nan")) for _, s in rows) / len(rows)
    print(f"{'mean':<10} {'':>3} {mean('PSNR'):>7.2f} {mean('SSIM'):>7.4f} {mean('LPIPS'):>7.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
