#!/usr/bin/env bash
# Fetch official 3D Gaussian Splatting at the commit RIDE-Splat was developed against, and build its
# CUDA extensions into the active Python environment (which needs PyTorch with CUDA already).
#
#   bash scripts/install_3dgs.sh                 # into third_party/gaussian-splatting
#   GS_ROOT=/path/to/checkout bash scripts/install_3dgs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="${GS_ROOT:-$ROOT/third_party/gaussian-splatting}"
COMMIT=54c035f7834b564019656c3e3fcc3646292f727d

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'PyTorch cannot see a CUDA GPU; install a CUDA build of PyTorch first.')"

if [ ! -d "$GS/.git" ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$GS"
fi
git -C "$GS" checkout -q "$COMMIT"
# simple-knn, diff-gaussian-rasterization (dr_aa) and fused-ssim; the SIBR viewer is not needed.
git -C "$GS" submodule update --init --recursive \
  submodules/simple-knn submodules/diff-gaussian-rasterization submodules/fused-ssim

pip install --no-build-isolation \
  "$GS/submodules/simple-knn" \
  "$GS/submodules/diff-gaussian-rasterization" \
  "$GS/submodules/fused-ssim"

echo "3D Gaussian Splatting $COMMIT is ready at $GS"
