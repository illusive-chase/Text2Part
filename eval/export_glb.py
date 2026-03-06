#!/usr/bin/env python3
"""Export valid objects as colored GLB files and zip them."""

import argparse
import colorsys
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_txt", default="valid.txt")
    parser.add_argument("--output_dir", default="eval/export")
    parser.add_argument("--zip_name", default="eval/export/valid_parts.zip")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = Path(args.valid_txt).read_text().strip().splitlines()
    print(f"Found {len(lines)} valid objects")

    glb_files = []
    for i, line in enumerate(lines):
        parts_dir = Path(line.strip())
        # extract name: outputs/gen4/faucet0/parts -> faucet0
        name = parts_dir.parent.name
        ply_files = sorted(glob.glob(str(parts_dir / "*.ply")))
        if not ply_files:
            print(f"  [{i+1}/{len(lines)}] {name}: no .ply files, skipping")
            continue

        print(f"  [{i+1}/{len(lines)}] {name} ({len(ply_files)} parts) ...", end="", flush=True)

        scene = trimesh.Scene()
        n_parts = len(ply_files)
        for idx, ply_path in enumerate(ply_files):
            mesh = trimesh.load(ply_path, force="mesh")
            hue = idx / n_parts
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            rgba = np.array([int(r*255), int(g*255), int(b*255), 255], dtype=np.uint8)
            colors = np.tile(rgba, (len(mesh.vertices), 1))
            mesh.visual.vertex_colors = colors
            part_name = os.path.basename(ply_path)
            scene.add_geometry(mesh, node_name=part_name)

        glb_path = out_dir / f"{name}.glb"
        scene.export(str(glb_path))
        glb_files.append(glb_path)
        print(f" ok")

    print(f"\nZipping {len(glb_files)} files ...")
    zip_path = Path(args.zip_name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for gf in glb_files:
            zf.write(str(gf), gf.name)
    print(f"Done: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
