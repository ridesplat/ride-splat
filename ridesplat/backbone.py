"""Locate the official 3D Gaussian Splatting code and make its loader split by image name.

RIDE-Splat trains on top of the official 3DGS implementation (graphdeco-inria/gaussian-splatting,
fetched at GS_COMMIT by scripts/install_3dgs.sh). Nothing here changes how 3DGS trains or renders.
"""
import os
import sys

GS_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
EVAL_PREFIX = "eval_"


def gs_root():
    """Path to the 3DGS checkout: $GS_ROOT, else third_party/gaussian-splatting in this repo."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.environ.get("GS_ROOT") or os.path.join(here, "third_party", "gaussian-splatting")
    if not os.path.isfile(os.path.join(root, "scene", "dataset_readers.py")):
        sys.exit(f"3D Gaussian Splatting not found at {root}. "
                 "Run scripts/install_3dgs.sh, or point GS_ROOT at a checkout.")
    return root


def add_gs_to_path():
    root = gs_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def split_by_prefix(cams, prefix=EVAL_PREFIX):
    """Split camera infos into (train, test): names starting with `prefix` are held out.

    Repeated image names are kept once, and both lists are sorted by name.
    """
    seen, unique = set(), []
    for cam in cams:
        if cam.image_name in seen:
            continue
        seen.add(cam.image_name)
        unique.append(cam)
    unique.sort(key=lambda cam: cam.image_name)
    train = [cam for cam in unique if not cam.image_name.startswith(prefix)]
    test = [cam for cam in unique if cam.image_name.startswith(prefix)]
    return train, test


def install_prefix_split(prefix=EVAL_PREFIX):
    """With --eval, hold out exactly the images whose names start with `prefix`.

    Stock 3DGS holds out every 8th image instead, which would put evaluation frames in the training
    set. Call this after add_gs_to_path() and before constructing a Scene.
    """
    import scene.dataset_readers as readers

    original = readers.readColmapSceneInfo
    if getattr(original, "_ridesplat_split", False):
        return

    def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
        info = original(path, images, depths, eval, train_test_exp, llffhold=llffhold)
        if not eval:
            return info
        train, test = split_by_prefix(list(info.train_cameras) + list(info.test_cameras), prefix)
        if not test:
            sys.exit(f"--eval was given but no image in {path} is named '{prefix}*'. RIDE-Splat holds "
                     f"out the {prefix}* frames; rename the held-out frames or drop --eval.")
        print(f"[split] {len(train)} training views, {len(test)} held-out views ({prefix}*)")
        return info._replace(train_cameras=train, test_cameras=test)

    readColmapSceneInfo._ridesplat_split = True
    readers.readColmapSceneInfo = readColmapSceneInfo
    readers.sceneLoadTypeCallbacks["Colmap"] = readColmapSceneInfo


def seed_everything(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
