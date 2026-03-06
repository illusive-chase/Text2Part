#!/usr/bin/env python3
"""
Manually add specified entries to eval/case_study/ without running VLM inference.

Exports GLB (same coloring logic as case_study.py) and copies render PNGs.
Does NOT write detection.json.
Skips entries whose output directory already exists.
"""

import colorsys
import shutil
from pathlib import Path

import numpy as np
import trimesh

RENDERS_DIR = Path("eval/renders")
OUTPUTS_DIR = Path("outputs/gen4")
CASE_STUDY_DIR = Path("eval/case_study")

ENTRIES = {
    "incomplete":        ["fridge2", "fridge4", "trash7", "trash13"],
    "complex_hinge":     ["stationery8"],
    "orientation_error": ["trash15"],
}


def export_glb(parts_dir: Path, glb_path: Path):
    ply_files = sorted(parts_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files in {parts_dir}")

    scene = trimesh.Scene()
    n = len(ply_files)
    for idx, ply in enumerate(ply_files):
        mesh = trimesh.load(str(ply), force="mesh")
        hue = idx / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        rgba = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=ply.name)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(glb_path))


def main():
    results = {}  # defect -> list of (name, status)

    for defect, names in ENTRIES.items():
        results[defect] = []
        for name in names:
            out_dir = CASE_STUDY_DIR / defect / name

            if out_dir.exists():
                print(f"  [{defect}] {name}: skipped (already exists)")
                results[defect].append((name, "skipped"))
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. export GLB
            parts_dir = OUTPUTS_DIR / name / "parts"
            glb_path = out_dir / f"{name}.glb"
            try:
                export_glb(parts_dir, glb_path)
                glb_status = "ok"
            except Exception as e:
                glb_status = f"FAILED ({e})"
                print(f"  [{defect}] {name}: GLB export failed: {e}")

            # 2. copy render PNGs
            render_src = RENDERS_DIR / name
            png_count = 0
            if render_src.exists():
                for png in sorted(render_src.glob("view_*.png")):
                    shutil.copy2(str(png), str(out_dir / png.name))
                    png_count += 1
            else:
                print(f"  [{defect}] {name}: render dir not found: {render_src}")

            print(f"  [{defect}] {name}: GLB={glb_status}, PNGs={png_count}")
            results[defect].append((name, f"GLB={glb_status} PNGs={png_count}"))

    # summary
    print("\n" + "=" * 60)
    print("case_study/ structure summary")
    print("=" * 60)
    for defect_dir in sorted(CASE_STUDY_DIR.iterdir()):
        if not defect_dir.is_dir():
            continue
        entries = sorted(defect_dir.iterdir())
        print(f"\n[{defect_dir.name}]  ({len(entries)} entries)")
        for entry in entries:
            if not entry.is_dir():
                continue
            files = list(entry.iterdir())
            glbs = [f for f in files if f.suffix == ".glb"]
            pngs = [f for f in files if f.suffix == ".png"]
            extras = [f for f in files if f.suffix not in (".glb", ".png")]
            parts = []
            if glbs:
                parts.append(f"glb={len(glbs)}")
            if pngs:
                parts.append(f"png={len(pngs)}")
            if extras:
                parts.append(f"other={[f.name for f in extras]}")
            print(f"  {entry.name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
