#!/usr/bin/env python3
"""
Batch pipeline: render all gen4 objects then run VLM judge.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --gen_dir outputs/gen4 --render_dir eval/renders --report eval/report.json
    python eval/evaluate.py --blender /path/to/blender --api_base http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

CATEGORIES = {
    "faucet", "fridge", "micro", "scissors",
    "stationery", "toolbox", "trash", "washing",
}

EVAL_DIR = Path(__file__).parent


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render + VLM judge pipeline for gen4 objects."
    )
    parser.add_argument(
        "--gen_dir", type=Path, default=Path("outputs/gen4"),
        help="Root directory of generated objects (default: outputs/gen4).",
    )
    parser.add_argument(
        "--render_dir", type=Path, default=Path("eval/renders"),
        help="Root directory for rendered outputs (default: eval/renders).",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("eval/report.json"),
        help="Output path for the summary report (default: eval/report.json).",
    )
    parser.add_argument(
        "--api_base", type=str, default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL for vlm_judge.py.",
    )
    parser.add_argument(
        "--objects", nargs="+", metavar="NAME",
        help="Only process these object names (e.g. --objects faucet0 faucet1). Default: all.",
    )
    return parser.parse_args()


# ── scanning ───────────────────────────────────────────────────────────────────

def extract_category(name: str) -> str | None:
    """Strip trailing digits from a directory name to recover the category."""
    cat = re.sub(r"\d+$", "", name)
    return cat if cat in CATEGORIES else None


def scan_gen_dir(gen_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Scan gen_dir for valid objects.

    Returns a sorted list of (name, category, parts_dir) tuples where:
      - name      : directory name, e.g. 'faucet0'
      - category  : e.g. 'faucet'
      - parts_dir : Path to the parts/ subdirectory containing .ply files
    """
    if not gen_dir.exists():
        print(f"[ERROR] gen_dir not found: {gen_dir}", file=sys.stderr)
        return []

    objects: list[tuple[str, str, Path]] = []
    for subdir in sorted(gen_dir.iterdir()):
        if not subdir.is_dir():
            continue
        category = extract_category(subdir.name)
        if category is None:
            continue
        parts_dir = subdir / "parts"
        if not parts_dir.is_dir():
            continue
        if not list(parts_dir.glob("*.ply")):
            continue
        objects.append((subdir.name, category, parts_dir))

    return objects


# ── subprocess runners ─────────────────────────────────────────────────────────

def run_render(name: str, parts_dir: Path, out_dir: Path) -> bool:
    """
    Call render_parts.py with the same Python interpreter (pip-installed bpy).
    Writes view_*.png and verify.json into out_dir.
    Returns True on success.
    """
    render_script = EVAL_DIR / "render_parts.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(render_script),
        "--parts_dir", str(parts_dir),
        "--output_dir", str(out_dir),
    ]
    print(f"  [render] {name} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [render FAILED] {name} (exit {proc.returncode})", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return False
    return True


def run_judge(name: str, category: str, render_dir: Path, api_base: str) -> bool:
    """
    Call vlm_judge.py with plain Python.
    Writes judge.json into render_dir.
    Returns True on success.
    """
    judge_script = EVAL_DIR / "vlm_judge.py"
    cmd = [
        sys.executable,
        str(judge_script),
        "--render_dir", str(render_dir),
        "--category", category,
        "--api_base", api_base,
    ]
    print(f"  [judge]  {name} ({category}) ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [judge FAILED] {name} (exit {proc.returncode})", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return False
    return True


# ── result loading ─────────────────────────────────────────────────────────────

def load_judge(render_dir: Path) -> dict | None:
    """Read judge.json; return None if missing or malformed."""
    p = render_dir / "judge.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def get_num_parts(render_dir: Path, parts_dir: Path) -> int:
    """
    Get num_parts from verify.json (written by render_parts.py).
    Falls back to counting .ply files in parts_dir.
    """
    verify = render_dir / "verify.json"
    if verify.exists():
        try:
            return int(json.loads(verify.read_text()).get("num_parts", 0))
        except (json.JSONDecodeError, ValueError):
            pass
    return len(list(parts_dir.glob("*.ply")))


# ── reporting ──────────────────────────────────────────────────────────────────

def generate_report(results: list[dict], report_path: Path) -> None:
    """Write eval/report.json."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"total": len(results), "results": results}, indent=2))
    print(f"\n[report] Written → {report_path}")


def print_summary_table(results: list[dict]) -> None:
    """Print a human-readable per-category summary table to stdout."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    COL = (14, 7, 18, 20)
    header = (
        f"{'Category':<{COL[0]}}"
        f"{'Count':>{COL[1]}}"
        f"{'Avg Object Score':>{COL[2]}}"
        f"{'Avg Fragment Score':>{COL[3]}}"
    )
    sep = "─" * (sum(COL) + len(COL) - 1)

    print("\n" + "═" * len(sep))
    print("EVALUATION SUMMARY")
    print("═" * len(sep))
    print(header)
    print(sep)

    all_obj: list[float] = []
    all_frag: list[float] = []

    for cat in sorted(by_cat):
        items = by_cat[cat]
        obj_scores  = [r["object_score"]   for r in items]
        frag_scores = [r["fragment_score"]  for r in items]
        avg_obj  = sum(obj_scores)  / len(obj_scores)
        avg_frag = sum(frag_scores) / len(frag_scores)
        all_obj.extend(obj_scores)
        all_frag.extend(frag_scores)
        print(
            f"{cat:<{COL[0]}}"
            f"{len(items):>{COL[1]}}"
            f"{avg_obj:>{COL[2]}.2f}"
            f"{avg_frag:>{COL[3]}.2f}"
        )

    print(sep)
    overall_obj  = sum(all_obj)  / len(all_obj)  if all_obj  else 0.0
    overall_frag = sum(all_frag) / len(all_frag) if all_frag else 0.0
    print(
        f"{'TOTAL':<{COL[0]}}"
        f"{len(results):>{COL[1]}}"
        f"{overall_obj:>{COL[2]}.2f}"
        f"{overall_frag:>{COL[3]}.2f}"
    )
    print("═" * len(sep))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. discover objects ───────────────────────────────────────────────────
    objects = scan_gen_dir(args.gen_dir)
    if not objects:
        print(f"No valid objects found in {args.gen_dir}", file=sys.stderr)
        sys.exit(1)

    if args.objects:
        allowed = set(args.objects)
        objects = [(n, c, p) for n, c, p in objects if n in allowed]
        if not objects:
            print(f"None of {args.objects} found in {args.gen_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(objects)} objects in {args.gen_dir}\n")

    results:  list[dict] = []
    n_skipped = 0
    n_failed  = 0

    # ── 2. process each object ────────────────────────────────────────────────
    for idx, (name, category, parts_dir) in enumerate(objects, 1):
        obj_render_dir = args.render_dir / name
        print(f"[{idx:>3}/{len(objects)}] {name}", flush=True)

        # Resume support: skip if judge.json already present
        if (obj_render_dir / "judge.json").exists():
            print(f"  [skip]   judge.json exists")
            judge = load_judge(obj_render_dir)
            if judge is not None:
                results.append({
                    "name":           name,
                    "category":       category,
                    "num_parts":      get_num_parts(obj_render_dir, parts_dir),
                    "object_score":   judge.get("object_score", 0),
                    "fragment_score": judge.get("fragment_score", 0),
                })
            n_skipped += 1
            continue

        # Step 1 — render
        if not run_render(name, parts_dir, obj_render_dir):
            n_failed += 1
            continue

        # Step 2 — judge
        if not run_judge(name, category, obj_render_dir, args.api_base):
            n_failed += 1
            continue

        # Collect result
        judge = load_judge(obj_render_dir)
        if judge is None:
            print(f"  [warn] judge.json missing after successful run", file=sys.stderr)
            n_failed += 1
            continue

        results.append({
            "name":           name,
            "category":       category,
            "num_parts":      get_num_parts(obj_render_dir, parts_dir),
            "object_score":   judge.get("object_score", 0),
            "fragment_score": judge.get("fragment_score", 0),
        })

    # ── 3. report ─────────────────────────────────────────────────────────────
    print(f"\nDone. Processed {len(results)} objects ({n_skipped} skipped, {n_failed} failed).")

    if not results:
        print("No results to report.", file=sys.stderr)
        sys.exit(1)

    generate_report(results, args.report)
    print_summary_table(results)


if __name__ == "__main__":
    main()
