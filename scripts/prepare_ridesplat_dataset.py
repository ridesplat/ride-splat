"""Download the RIDE-Splat Dataset (optional) and lay each scene out for 3D Gaussian Splatting.

The release keeps the reflective training frames in train/ and the held-out frames in eval/. 3DGS
reads every image from one images/ folder, so this links both there and checks every image the
COLMAP model refers to is present.

    python scripts/prepare_ridesplat_dataset.py --root data/ride-splat-dataset --download
"""
import os
import shutil
import struct
import sys
import urllib.request
import zipfile
from argparse import ArgumentParser

RELEASE_URL = ("https://github.com/ridesplat/ride-splat/releases/download/"
               "dataset-v0.1/ride-splat-dataset-v0.1.zip")


def colmap_image_names(images_bin):
    """Image names stored in a COLMAP images.bin."""
    names = []
    with open(images_bin, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            f.read(4 + 32 + 24 + 4)                    # image id, rotation, translation, camera id
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_points,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_points)                     # x, y, point3D id per keypoint
            names.append(name.decode())
    return names


def prepare_scene(scene, copy=False):
    images = os.path.join(scene, "images")
    os.makedirs(images, exist_ok=True)
    for split in ("train", "eval"):
        for f in sorted(os.listdir(os.path.join(scene, split))):
            dst = os.path.join(images, f)
            if os.path.lexists(dst):
                continue
            if copy:
                shutil.copy2(os.path.join(scene, split, f), dst)
            else:
                os.symlink(os.path.join("..", split, f), dst)
    names = colmap_image_names(os.path.join(scene, "sparse", "0", "images.bin"))
    missing = [n for n in names if not os.path.exists(os.path.join(images, n))]
    if missing:
        sys.exit(f"{scene}: {len(missing)} image(s) in the COLMAP model are missing, e.g. {missing[:3]}")
    n_eval = sum(n.startswith("eval_") for n in names)
    print(f"{os.path.basename(scene)}: {len(names) - n_eval} training views, {n_eval} held-out views")


def download(root):
    """Fetch the release zip and unpack its scenes into `root`."""
    parent = os.path.dirname(os.path.abspath(root))
    os.makedirs(parent, exist_ok=True)
    archive = os.path.join(parent, "ride-splat-dataset-v0.1.zip")
    if not os.path.exists(archive):
        print(f"downloading {RELEASE_URL}")
        urllib.request.urlretrieve(RELEASE_URL, archive + ".part")
        os.replace(archive + ".part", archive)
    with zipfile.ZipFile(archive) as z:
        top = {n.split("/")[0] for n in z.namelist()}
        z.extractall(parent)
    if top != {os.path.basename(os.path.abspath(root))}:
        sys.exit(f"unexpected archive layout {sorted(top)}; expected a single {os.path.basename(root)}/ folder")


def main(argv=None):
    ap = ArgumentParser(description="Prepare the RIDE-Splat Dataset for training")
    ap.add_argument("--root", default="data/ride-splat-dataset")
    ap.add_argument("--download", action="store_true", help="download and unpack the dataset (676 MB) first")
    ap.add_argument("--copy", action="store_true", help="copy frames into images/ instead of linking them")
    args = ap.parse_args(argv)
    if args.download:
        download(args.root)
    scenes = sorted(os.path.join(args.root, d) for d in os.listdir(args.root)
                    if os.path.isfile(os.path.join(args.root, d, "sparse", "0", "images.bin")))
    if not scenes:
        sys.exit(f"No scenes (folders with sparse/0/images.bin) under {args.root}.")
    for scene in scenes:
        prepare_scene(scene, copy=args.copy)


if __name__ == "__main__":
    main()
