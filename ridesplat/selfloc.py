"""Reflection self-localization: a fixed spatial weight W built from the pass-1 reflection image.

W marks where the reflection lives. Pass 2 multiplies R by W, so pixels with little reflection
evidence must be explained by the 3DGS scene instead (paper, Sec. 3.2).
"""
import numpy as np
import torch
from PIL import Image


def weight_from_reflection(R, sigma=8.0):
    """W from a signed reflection image R [3,H,W]: per-pixel max |R| over channels, scaled to [0,1],
    then smoothed by a Gaussian with standard deviation `sigma` pixels. Returns float32 [H,W]."""
    from scipy.ndimage import gaussian_filter

    a = torch.as_tensor(R).float().abs()
    while a.dim() > 2:
        a = a.amax(dim=0)
    a = a.numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return np.clip(gaussian_filter(a, sigma), 0, 1)


def save_weight(W, path):
    Image.fromarray((np.clip(W, 0, 1) * 255).astype("uint8")).save(path)


def load_weight(path, size=None):
    """Load a weight PNG as a [1,H,W] tensor rescaled to span [0,1]; resized (nearest) to `size`."""
    w = np.asarray(Image.open(path).convert("L")).astype("float32")
    w = (w - w.min()) / (w.max() - w.min() + 1e-6)
    w = torch.from_numpy(w)[None]
    if size is not None and tuple(w.shape[-2:]) != tuple(size):
        w = torch.nn.functional.interpolate(w[None], size=tuple(size), mode="nearest")[0]
    return w
