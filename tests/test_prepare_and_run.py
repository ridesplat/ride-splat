import os
import struct
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import prepare_ridesplat_dataset as prep  # noqa: E402

import run_ridesplat  # noqa: E402


def write_images_bin(path, names):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names):
            f.write(struct.pack("<i", i + 1) + b"\0" * 56 + struct.pack("<i", 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 1) + b"\0" * 24)


def make_scene(root, train, held_out):
    scene = root / "rs1"
    for split, names in (("train", train), ("eval", held_out)):
        (scene / split).mkdir(parents=True)
        for n in names:
            Image.new("RGB", (4, 4)).save(scene / split / n)
    (scene / "sparse" / "0").mkdir(parents=True)
    write_images_bin(scene / "sparse" / "0" / "images.bin", train + held_out)
    return scene


def test_colmap_names_and_images_folder(tmp_path):
    scene = make_scene(tmp_path, ["train_r00000.png", "train_r00001.png"], ["eval_c00000.png"])
    assert prep.colmap_image_names(scene / "sparse" / "0" / "images.bin") == [
        "train_r00000.png", "train_r00001.png", "eval_c00000.png"]
    prep.prepare_scene(str(scene))
    assert sorted(os.listdir(scene / "images")) == ["eval_c00000.png", "train_r00000.png", "train_r00001.png"]
    assert Image.open(scene / "images" / "eval_c00000.png").size == (4, 4)


def test_missing_frame_is_an_error(tmp_path):
    scene = make_scene(tmp_path, ["train_r00000.png"], ["eval_c00000.png"])
    os.remove(scene / "eval" / "eval_c00000.png")
    with pytest.raises(SystemExit):
        prep.prepare_scene(str(scene))


def test_pass_commands_paper_and_minimal():
    args, extra = run_ridesplat.parse(["-s", "scene", "-m", "out", "--refl_l1", "0.002"])
    pass1, pass2 = run_ridesplat.pass_commands(args, extra)
    assert "--refl_weight" not in pass1 and "--atten" not in pass1
    assert pass1[-2:] == ["--refl_l1", "0.002"] and pass2[-2:] == ["--refl_l1", "0.002"]
    for flag in ("--refl_weight", "--render_dir", "--atten", "--appearance"):
        assert flag in pass2
    args, extra = run_ridesplat.parse(["-s", "scene", "-m", "out", "--minimal"])
    _, pass2 = run_ridesplat.pass_commands(args, extra)
    assert "--refl_weight" in pass2 and "--atten" not in pass2 and "--appearance" not in pass2
