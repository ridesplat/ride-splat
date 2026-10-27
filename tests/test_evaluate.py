import math

import numpy as np
from PIL import Image

import evaluate


def write(path, array):
    Image.fromarray(array).save(path)


def test_scores_pairs_by_name(tmp_path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    rng = np.random.default_rng(0)
    clean = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
    noisy = np.clip(clean.astype(int) + rng.integers(-8, 9, clean.shape), 0, 255).astype(np.uint8)
    write(gt / "eval_c00000.png", clean)
    write(gt / "eval_c00001.png", clean)
    write(pred / "eval_c00000.png", clean)
    write(pred / "eval_c00001.png", noisy)

    result = evaluate.evaluate(str(pred), str(gt), use_lpips=False)
    exact, off = result["per_frame"]
    assert exact["PSNR"] == 99.0 and math.isclose(exact["SSIM"], 1.0, abs_tol=1e-6)
    mse = np.mean((noisy.astype(np.float32) / 255 - clean.astype(np.float32) / 255) ** 2)
    assert math.isclose(off["PSNR"], 10 * math.log10(1 / mse), rel_tol=1e-5)
    assert result["summary"]["n"] == 2


def test_doubled_extension_still_pairs(tmp_path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    img = np.zeros((40, 48, 3), dtype=np.uint8)
    write(gt / "eval_c00000.png", img)
    write(pred / "eval_c00000.png.png", img)
    assert evaluate.evaluate(str(pred), str(gt), use_lpips=False)["summary"]["n"] == 1
