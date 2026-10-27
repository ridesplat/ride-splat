# RIDE-Splat: Reflection Removal for In-the-Wild Driving Environments

Kevin David Hayes, Alex Hanson, Allen Tu, Pranav Asthana, Matthias Zwicker, Tom Goldstein
University of Maryland, College Park

[Paper](https://arxiv.org/abs/2610.34632) | [Project page](https://ridesplat.github.io) | [Dataset](https://github.com/ridesplat/ride-splat-dataset)

Driving video recorded through a windshield carries reflections of the cabin, and 3D Gaussian
Splatting (3DGS) reconstructs them as part of the scene. RIDE-Splat represents the road with a 3DGS
scene and the dominant camera-attached reflection with one signed RGB image shared by every frame,
then drops that image at render time. It needs only posed video with reflections: no clean images,
reflection masks or pretrained networks.

## Setup

RIDE-Splat needs a CUDA GPU. It runs on the official
[3DGS code](https://github.com/graphdeco-inria/gaussian-splatting), which the install script fetches at
the commit RIDE-Splat was developed against (`54c035f`).

```
conda create -n ridesplat python=3.10 -y
conda activate ridesplat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
bash scripts/install_3dgs.sh          # set GS_ROOT to use an existing 3DGS checkout instead
```

## Data

```
python scripts/prepare_ridesplat_dataset.py --root data/ride-splat-dataset --download
```

This downloads the [RIDE-Splat Dataset](https://github.com/ridesplat/ride-splat-dataset) (676 MB):
three real driving scenes, `rs1` (Leesburg), `rs2` (Martin) and `rs3` (Reston). Each has reflective
training frames in `train/`, held-out frames from a separate non-reflective drive of the same route in
`eval/`, and COLMAP poses for both in `sparse/0/`. The script links `train/` and `eval/` into the single
`images/` folder 3DGS reads from.

## Run

```
python run_ridesplat.py -s data/ride-splat-dataset/rs2 -m output/rs2 --gt data/ride-splat-dataset/rs2/eval
```

| Step | Output in `output/rs2/` |
|---|---|
| Pass 1: fit the 3DGS scene and the shared reflection image R together | `pass1/`, `pass1/refl_layer.png` |
| Self-localization: turn \|R\| into a smooth spatial weight W | `weight.png` |
| Pass 2: refit from scratch with W fixed | `pass2/` |
| Render the held-out views from the scene alone | `renders/` |
| Score against the clean targets | `metrics.json` |

Rerunning the same command skips finished steps. `--smoke_test` runs 300-iteration passes to check an
installation in a few minutes. Any other `train.py` flag, such as `--refl_l1 0.002`, is passed to both
passes. Training uses frames named `train_*` and holds out every frame named `eval_*`.

## Configuration used in the paper

Every number reported in the paper uses the defaults below. Pass 2 also fits two terms that
`train.py` applies only during training, to the scene render:

* a per-pixel attenuation A in [0, 0.9), with an L1 penalty of 0.01 (`--atten --atten_max 0.9`)
* a per-frame RGB gain and bias that absorbs exposure and white-balance changes between frames, pulled
  toward identity with weight 10 (`--appearance affine`)

`run_ridesplat.py` enables both by default. `--minimal` runs pass 2 with the self-localization weight
only. Neither term is used when rendering held-out views.

| Setting | Value | Flag |
|---|---|---|
| Iterations per pass | 30,000 | `--iterations` |
| Resolution | full (1900x1025 on the RIDE-Splat Dataset) | `-r 1` |
| Learning rate of R | 0.01 | `--refl_lr` |
| L1 weight on R | 0.01 | `--refl_l1` |
| Warm-up before R is used | 2,000 iterations | `--refl_warmup` |
| Self-localization smoothing | 8 px | `--sigma` |
| Seed | 0 | `--seed` |

## Reproducing the RIDE-Splat Dataset results

```
bash scripts/reproduce_ridesplat_dataset.sh
```

This runs all three scenes (two 30k-iteration passes each, roughly 1.5 to 3 hours per scene on one
A100) and prints them next to the paper's numbers:

| Scene | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Leesburg (`rs1`) | 16.03 | 0.6070 | 0.5712 |
| Martin (`rs2`) | 20.18 | 0.8420 | 0.4025 |
| Reston (`rs3`) | 19.21 | 0.7794 | 0.4880 |
| Mean | 18.48 | 0.7428 | 0.4873 |

Expect small differences. 3DGS training on a GPU is not bit-for-bit deterministic, and the released
frames have faces and licence plates blurred, while the paper's numbers were computed before blurring.

## Other datasets

The paper also evaluates on XLD and on WayveScenes101 with synthetic reflections. Scripts that build those two benchmarks are not included yet. RIDE-Splat runs on
any COLMAP scene whose held-out frames are named `eval_*`: pass `--gt` a folder of clean targets with
the same file names.

## Evaluation

`evaluate.py` pairs renders and targets by file name and reports per-frame means of PSNR, SSIM
(11-tap Gaussian window, sigma 1.5) and LPIPS (VGG), on the full frame with no color correction.

## Tests

```
pytest
```

The tests run on a CPU and do not need 3DGS.

## License

`train.py` is adapted from 3DGS and, like 3DGS itself, is for non-commercial research and evaluation
use only. The other files are MIT-licensed. See [LICENSE.md](LICENSE.md).

## Citation

```bibtex
@article{hayes2026ridesplat,
  title   = {{RIDE-Splat}: Reflection Removal for In-the-Wild Driving Environments},
  author  = {Hayes, Kevin David and Hanson, Alex and Tu, Allen and Asthana, Pranav and Zwicker, Matthias and Goldstein, Tom},
  journal = {arXiv preprint arXiv:2610.34632},
  year    = {2026}
}
```
