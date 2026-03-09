# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text2Part is a text-to-3D-parts pipeline that generates articulated 3D objects from text prompts. It decomposes generated meshes into movable parts (doors, drawers, lids, handles, etc.). The pipeline has four stages, each in its own module:

1. **t2i** (Text→Image): Generates an image from a text prompt using Qwen-Image-2512 via diffusers
2. **i2m** (Image→Mesh): Converts the image to a 3D mesh using Hunyuan3D-2.1
3. **i2t** (Image→Texture): Applies textures to a mesh using Hunyuan3D Paint (not part of the main pipeline; separate utility)
4. **m2p** (Mesh→Parts): Decomposes the mesh into separate parts using Hunyuan3D-Part/XPart/PartFormer

## Running the Pipeline

Each stage is a standalone script invoked via `python <module>/main.py` with `tyro` CLI args:

```bash
# Full pipeline: text → image → mesh → parts
python t2i/main.py --prompt "A faucet with a handle. Background: white." --output outputs/gen/faucet/img.png
python i2m/main.py --image outputs/gen/faucet/img.png --output outputs/gen/faucet/mesh.ply
python m2p/main.py --mesh outputs/gen/faucet/mesh.ply --output outputs/gen/faucet/parts

# Parts-only from existing mesh (supports .glb, .ply, .obj)
python m2p/main.py --mesh data/fridge.glb --output outputs/obja/fridge
```

Parts are exported as individual `.ply` files in the output directory (one per part).

## Batch Generation

`scripts/` contains per-category batch scripts (faucet.sh, fridge.sh, etc.) that generate 16 variants (seeds 0-15) with the t2i→i2m→m2p pipeline. Run them via `scripts/all.sh`.

Categories: faucet, fridge, micro (microwave), scissors, stationery, toolbox, trash, washing.

## Visualization and Annotation

- **`m2p/vis.py`**: Visualize parts in browser via viser (port 6789). Accepts a directory of `.ply` files or a `.glb`.
- **`m2p/annotate.py`**: Interactive annotation tool (viser, port 6789). Accepts a directory of `.ply` files or a `.glb`/`.gltf`. Features:
  - **Part selection & merge**: Click parts to select, merge 2+ selected parts (keeps lowest name), undo merges.
  - **Hinge annotation**: Select exactly 2 parts to auto-detect a hinge axis via PCA on boundary vertices (larger mesh = base, smaller = child). Stores axis, pivot, and angle limits.
  - **Hinge preview**: Interactive slider rotates the child part around the detected axis in the viewer.
  - **URDF generation**: On export, if hinges are defined, generates `articulated.urdf` with revolute joints and OBJ mesh references.
  - **Export**: Writes `annotated.glb`, `annotation.json` (parts list, merge history, hinge definitions), and optionally `articulated.urdf`.

## Evaluation (`eval/`)

- **`render_parts.py`**: Headless Blender (via pip-installed `bpy`) renders 8 views of part-decomposed objects using Cycles CPU. Outputs `view_*.png` + `verify.json`.
- **`vlm_judge.py`**: Uses an OpenAI-compatible VLM API to judge object quality and detect floating fragments. Requires a running vLLM server (default `http://127.0.0.1:8000/v1`).
- **`evaluate.py`**: Orchestrates render + judge for batch evaluation. Writes `report.json` with per-category scores.
- **`case_study.py`**: Uses VLM to detect defects (incomplete, support_structure, under_seg, complex_hinge). Exports top candidates as GLB + renders.
- **`export_glb.py`**: Exports valid objects as colored GLB files and zips them.

## Third-Party Dependencies (Git Submodules)

- `third_party/hunyuan3d` — Fork of Hunyuan3D-2.1 (shape generation + texture painting)
- `third_party/hunyuan3dpart` — Fork of Hunyuan3D-Part (XPart partformer + P3-SAM)

Pipeline scripts add these to `sys.path` at runtime (e.g., `sys.path.insert(0, './third_party/hunyuan3d/hy3dshape')`).

## Environment

- Python 3.11, venv at `.venv/`
- Requires CUDA GPU for inference
- Key deps: torch 2.7, diffusers (git head), trimesh, tyro, viser, bpy, open3d, pytorch-lightning, flash_attn
- Formatter: yapf (via VS Code extension `eeyore.yapf`), with ruff for linting/isort
- Line length: 120 chars, single quotes, LF line endings

## Code Conventions

- Each pipeline module has a `@dataclass` class with `init(device)` and `inference(...)` methods
- CLI entry points use `tyro.cli(main)` pattern
- All scripts assume execution from the repo root (`python t2i/main.py ...`)
- Output directories are auto-created with `Path.mkdir(exist_ok=True, parents=True)`
