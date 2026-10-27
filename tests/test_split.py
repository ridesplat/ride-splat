import sys
import types
from typing import NamedTuple

import pytest

from ridesplat import backbone


class Cam(NamedTuple):
    image_name: str


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list


def test_frame_stem_drops_the_image_extension():
    assert backbone.frame_stem("eval_c00000.png") == "eval_c00000"
    assert backbone.frame_stem("eval_c00000") == "eval_c00000"
    assert backbone.frame_stem("frame.v2.JPG") == "frame.v2"


def test_split_by_prefix_holds_out_eval_frames():
    cams = [Cam("train_r00001"), Cam("eval_c00000"), Cam("train_r00000"), Cam("eval_c00000")]
    train, test = backbone.split_by_prefix(cams)
    assert [c.image_name for c in train] == ["train_r00000", "train_r00001"]
    assert [c.image_name for c in test] == ["eval_c00000"]


@pytest.fixture
def fake_3dgs(monkeypatch):
    """Stand-in for 3DGS's scene.dataset_readers, which splits every 8th image like stock 3DGS."""
    names = [f"train_r{i:05d}" for i in range(12)] + [f"eval_c{i:05d}" for i in range(3)]

    def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
        cams = [Cam(n) for n in sorted(names)]
        test = [c for i, c in enumerate(cams) if eval and i % llffhold == 0]
        return SceneInfo([c for c in cams if c not in test], test)

    readers = types.ModuleType("scene.dataset_readers")
    readers.readColmapSceneInfo = readColmapSceneInfo
    readers.sceneLoadTypeCallbacks = {"Colmap": readColmapSceneInfo}
    package = types.ModuleType("scene")
    package.dataset_readers = readers
    monkeypatch.setitem(sys.modules, "scene", package)
    monkeypatch.setitem(sys.modules, "scene.dataset_readers", readers)
    return readers, names


def test_loader_holds_out_exactly_the_eval_frames(fake_3dgs):
    readers, names = fake_3dgs
    backbone.install_prefix_split()
    info = readers.sceneLoadTypeCallbacks["Colmap"]("scene", "images", "", True, False)
    assert sorted(c.image_name for c in info.test_cameras) == [n for n in sorted(names) if n.startswith("eval_")]
    assert not any(c.image_name.startswith("eval_") for c in info.train_cameras)
    assert len(info.train_cameras) == 12


def test_loader_refuses_eval_without_eval_frames(fake_3dgs, monkeypatch):
    readers, names = fake_3dgs
    names[:] = [n for n in names if not n.startswith("eval_")]
    backbone.install_prefix_split()
    with pytest.raises(SystemExit):
        readers.sceneLoadTypeCallbacks["Colmap"]("scene", "images", "", True, False)
