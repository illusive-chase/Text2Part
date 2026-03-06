#!/usr/bin/env python3
"""
Case study: use a local vLLM (Qwen3-VL-8B-Instruct) to detect defects in
part-decomposed 3D object renders, then export examples for each category.

Defect categories:
  incomplete       - severely incomplete or chaotic/wrong shape
  support_structure - 3D-printing support base plate visible at bottom
  under_seg        - under-segmented parts (would move wrong parts)
  complex_hinge    - overly complex hinge relationships (telescoping, linkages)
"""

import base64
import colorsys
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests
import trimesh

# ── paths ────────────────────────────────────────────────────────────────────
RENDERS_DIR = Path("eval/renders")
OUTPUTS_DIR = Path("outputs/gen4")
RESULTS_JSON = Path("eval/case_study_results.json")
CASE_STUDY_DIR = Path("eval/case_study")
SUMMARY_TXT = Path("/tmp/case_study_summary.txt")

VLLM_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

DEFECT_CATEGORIES = ["incomplete", "support_structure", "under_seg", "complex_hinge"]
TOP_K = 9  # candidates per category to export (8-10)


# ── proxy scrub ──────────────────────────────────────────────────────────────
def clear_proxies():
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
        os.environ.pop(key, None)


# ── VLM helpers ──────────────────────────────────────────────────────────────
def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


SYSTEM_PROMPT = """You are an expert 3D-model quality inspector.
You will be shown multiple rendered views of a single 3D object that has been decomposed into colored parts.
Analyze the object carefully and return a JSON object with exactly these fields:

{
  "incomplete":        {"detected": true/false, "confidence": "high"/"medium"/"low", "reason": "..."},
  "support_structure": {"detected": true/false, "confidence": "high"/"medium"/"low", "reason": "..."},
  "under_seg":         {"detected": true/false, "confidence": "high"/"medium"/"low", "reason": "..."},
  "complex_hinge":     {"detected": true/false, "confidence": "high"/"medium"/"low", "reason": "..."}
}

Definitions:
- incomplete: The object is severely incomplete, has chaotic geometry, or does not match the expected object shape at all.
- support_structure: A flat base plane / base plate at the bottom is visible — this is a 3D printing support artifact, not part of the real object.
- under_seg: Some parts that should be separate (e.g., a door and its frame) are merged into one colored region, so articulation would move wrong geometry.
- complex_hinge: The object has overly complex hinge relationships such as telescoping lid supports, scissor mechanisms, or multi-link transmission linkages that cannot be simply annotated.

Return ONLY the raw JSON object, no markdown fences, no extra text."""


def query_vlm(image_paths: list[Path], object_name: str, category: str) -> dict:
    """Call the vLLM chat endpoint with all 8 view images and return parsed JSON."""
    clear_proxies()

    content = []
    for img_path in image_paths:
        b64 = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    content.append({
        "type": "text",
        "text": (
            f"The following {len(image_paths)} images show different views of a "
            f"'{category}' 3D object named '{object_name}'. "
            "Each part is colored differently. Inspect all views and output the defect JSON."
        )
    })

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    resp = requests.post(
        f"{VLLM_BASE}/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    return json.loads(raw)


# ── GLB export ───────────────────────────────────────────────────────────────
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


# ── scoring helper ────────────────────────────────────────────────────────────
CONF_SCORE = {"high": 3, "medium": 2, "low": 1}


def defect_score(detection: dict) -> int:
    if not detection.get("detected", False):
        return 0
    return CONF_SCORE.get(detection.get("confidence", "low"), 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    clear_proxies()

    # discover objects that have renders
    verify_files = sorted(RENDERS_DIR.glob("*/verify.json"))
    print(f"Found {len(verify_files)} objects with renders")

    # load cached results if they exist (so we can resume)
    if RESULTS_JSON.exists():
        all_results = json.loads(RESULTS_JSON.read_text())
        print(f"Loaded {len(all_results)} cached results from {RESULTS_JSON}")
    else:
        all_results = {}

    # ── VLM inference loop ────────────────────────────────────────────────────
    total = len(verify_files)
    for i, vf in enumerate(verify_files):
        name = vf.parent.name
        if name in all_results:
            print(f"  [{i+1}/{total}] {name}: cached, skipping")
            continue

        # derive object category (strip trailing digits)
        cat = name.rstrip("0123456789")

        verify = json.loads(vf.read_text())
        image_files = verify.get("image_files", [])
        image_paths = [vf.parent / f for f in image_files]
        image_paths = [p for p in image_paths if p.exists()]

        if not image_paths:
            print(f"  [{i+1}/{total}] {name}: no images found, skipping")
            continue

        print(f"  [{i+1}/{total}] {name} ({len(image_paths)} views) ...", end="", flush=True)
        try:
            result = query_vlm(image_paths, name, cat)
            all_results[name] = {"category": cat, "detections": result}
            print(" ok")
        except Exception as e:
            print(f" ERROR: {e}")
            all_results[name] = {"category": cat, "detections": None, "error": str(e)}

        # save after every object so we can resume
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(json.dumps(all_results, indent=2))

        # small delay to be nice to the local server
        time.sleep(0.2)

    # ── select top candidates per defect category ─────────────────────────────
    print("\nSelecting top candidates per defect category ...")

    selected: dict[str, list] = {d: [] for d in DEFECT_CATEGORIES}

    for name, info in all_results.items():
        dets = info.get("detections")
        if not dets:
            continue
        for defect in DEFECT_CATEGORIES:
            det = dets.get(defect, {})
            score = defect_score(det)
            if score > 0:
                selected[defect].append((score, name, info["category"], det))

    # sort by score desc, then by object category for diversity
    candidates: dict[str, list] = {}
    for defect in DEFECT_CATEGORIES:
        items = sorted(selected[defect], key=lambda x: -x[0])
        # take up to TOP_K, trying to keep category diversity
        seen_cats: dict[str, int] = {}
        picked = []
        for score, name, cat, det in items:
            if len(picked) >= TOP_K:
                break
            # allow at most 3 from same object category (for diversity)
            if seen_cats.get(cat, 0) >= 3:
                continue
            picked.append((name, cat, det, score))
            seen_cats[cat] = seen_cats.get(cat, 0) + 1
        candidates[defect] = picked
        print(f"  {defect}: {len(picked)} candidates")

    # ── export selected candidates ────────────────────────────────────────────
    print("\nExporting GLBs and renders ...")

    export_summary: dict[str, list] = {d: [] for d in DEFECT_CATEGORIES}

    for defect, items in candidates.items():
        for name, cat, det, score in items:
            out_dir = CASE_STUDY_DIR / defect / name
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. export GLB
            parts_dir = OUTPUTS_DIR / name / "parts"
            glb_path = out_dir / f"{name}.glb"
            try:
                export_glb(parts_dir, glb_path)
                glb_ok = True
            except Exception as e:
                print(f"    GLB export failed for {name}: {e}")
                glb_ok = False

            # 2. copy render PNGs
            render_src = RENDERS_DIR / name
            png_count = 0
            for png in sorted(render_src.glob("view_*.png")):
                shutil.copy2(str(png), str(out_dir / png.name))
                png_count += 1

            # 3. save detection metadata
            meta = {
                "name": name,
                "category": cat,
                "defect": defect,
                "score": score,
                "detection": det,
                "glb_exported": glb_ok,
                "pngs_copied": png_count,
            }
            (out_dir / "detection.json").write_text(json.dumps(meta, indent=2))

            export_summary[defect].append(name)
            conf = det.get("confidence", "?")
            reason = det.get("reason", "")[:80]
            print(f"    [{defect}] {name} (score={score}, conf={conf}): {reason}")

    # ── write summary ─────────────────────────────────────────────────────────
    lines = ["Case Study Results", "=" * 60, ""]
    total_detections = {d: len(selected[d]) for d in DEFECT_CATEGORIES}
    for defect in DEFECT_CATEGORIES:
        lines.append(f"[{defect}]")
        lines.append(f"  Total detected: {total_detections[defect]}")
        lines.append(f"  Exported ({len(export_summary[defect])}):")
        for name in export_summary[defect]:
            det = all_results[name]["detections"][defect]
            conf = det.get("confidence", "?")
            reason = det.get("reason", "")[:100]
            lines.append(f"    {name} ({conf}): {reason}")
        lines.append("")

    lines.append(f"Full results: {RESULTS_JSON.resolve()}")
    lines.append(f"Exports:      {CASE_STUDY_DIR.resolve()}")

    summary_text = "\n".join(lines)
    SUMMARY_TXT.write_text(summary_text)
    print(f"\nSummary written to {SUMMARY_TXT}")
    print(summary_text)


if __name__ == "__main__":
    main()
