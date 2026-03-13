"""
Auto-merge over-segmented parts using a VLM (Claude) to analyze rendered views.

Renders 8 views of the colored parts, sends them to the VLM with a color legend,
and applies the merge/delete decisions to produce a cleaned-up set of parts.

Usage:
    python m2p/auto_merge.py --parts outputs/gen4/faucet0/parts --category faucet
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import trimesh
import tyro

# ── view file names produced by render_parts.py (8 azimuths, 45° apart) ──
VIEW_NAMES = [f'view_{az:03d}.png' for az in range(0, 360, 45)]

# Named colors for the legend — ordered roughly by hue (red → magenta)
NAMED_COLORS = [
    ('Red', (1.0, 0.0, 0.0)),
    ('Orange', (1.0, 0.5, 0.0)),
    ('Yellow', (1.0, 1.0, 0.0)),
    ('Lime', (0.5, 1.0, 0.0)),
    ('Green', (0.0, 1.0, 0.0)),
    ('Teal', (0.0, 1.0, 0.5)),
    ('Cyan', (0.0, 1.0, 1.0)),
    ('Sky Blue', (0.0, 0.5, 1.0)),
    ('Blue', (0.0, 0.0, 1.0)),
    ('Purple', (0.5, 0.0, 1.0)),
    ('Magenta', (1.0, 0.0, 1.0)),
    ('Pink', (1.0, 0.0, 0.5)),
]


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (all in [0,1]) to RGB (all in [0,1])."""
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    return [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][i]


def nearest_color_name(r: float, g: float, b: float) -> str:
    """Return the name of the nearest predefined color."""
    best_name = 'Unknown'
    best_dist = float('inf')
    for name, (cr, cg, cb) in NAMED_COLORS:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def build_legend(part_names: list[str]) -> str:
    """Build a text legend mapping part labels to their rendered colors."""
    n = len(part_names)
    lines = []
    for idx, name in enumerate(part_names):
        hue = idx / n
        r, g, b = hsv_to_rgb(hue, 0.9, 0.9)
        hex_color = f'#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}'
        color_name = nearest_color_name(r, g, b)
        lines.append(f'Part {name}: {color_name} ({hex_color})')
    return '\n'.join(lines)


USER_PROMPT = """\
You are analyzing a 3D object that has been automatically decomposed into colored \
parts. The decomposition is over-segmented — many parts that should be a single \
rigid component are split into multiple pieces.

{category_line}

Your task is to determine which parts should be MERGED and which should be DELETED, \
so that the final parts represent semantically meaningful rigid bodies with clear \
articulation relationships between them.

GOALS:
1. Each final part should be a RIGID BODY — a component that moves as one unit
2. Between any two adjacent final parts, there should be a clear ARTICULATION \
relationship (hinge joint, rotation, or sliding)
3. Delete small disconnected floating fragments or artifacts
4. Minimize the number of final parts while preserving all real articulation joints

EXAMPLES of correct decomposition:
- Fridge: body (frame+shelves merged) + door → 1 hinge
- Faucet: body + handle → 1 rotation joint
- Scissors: blade_A + blade_B → 1 pivot joint
- Microwave: body + door + handle → door hinges on body, handle is rigid with door \
(so handle merges into door)

COLOR LEGEND (each part has a consistent color across all 8 views):
{legend}

The 8 images below show the object from evenly-spaced angles.

Respond with EXACTLY this JSON (no other text):
{{
  "merge": [["00", "02", "05"], ["01", "03"]],
  "delete": ["06"],
  "parts_description": {{
    "00+02+05": "main body/frame",
    "01+03": "door panel"
  }},
  "reasoning": "Brief explanation"
}}

RULES:
- "merge": groups of 2+ parts to combine into one rigid body
- "delete": parts to remove (floating fragments, tiny artifacts)
- Parts NOT listed in any merge group or delete list are kept as-is
- Every part label must appear AT MOST once across all merge groups and delete list
- Use the part labels from the color legend (e.g., "00", "01")
- If no merges or deletes are needed, use empty arrays"""


def parse_vlm_json(raw: str) -> dict:
    """Extract JSON from VLM response, tolerant of markdown fences."""
    text = raw.strip()
    if '```' in text:
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
    return json.loads(text)


def render_parts(parts_dir: Path, render_dir: Path) -> None:
    """Call eval/render_parts.py as a subprocess to render 8 views."""
    render_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, 'eval/render_parts.py',
        '--parts_dir', str(parts_dir),
        '--output_dir', str(render_dir),
    ]
    print(f'[render] {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[render] stdout:\n{result.stdout}')
        print(f'[render] stderr:\n{result.stderr}')
        raise RuntimeError(f'render_parts.py failed with exit code {result.returncode}')
    print(result.stdout)


def call_vlm(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
) -> str:
    """Send images + prompt to the Anthropic API and return the raw response text."""
    import anthropic

    client = anthropic.Anthropic()

    content: list[dict] = []
    content.append({'type': 'text', 'text': prompt_text})
    for img_path in image_paths:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/png',
                'data': b64,
            },
        })

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        messages=[{'role': 'user', 'content': content}],
    )
    return response.content[0].text


def validate_response(decision: dict, valid_names: set[str]) -> tuple[list[list[str]], list[str]]:
    """Validate and clean the VLM response. Returns (merge_groups, delete_list)."""
    merge_groups: list[list[str]] = []
    delete_list: list[str] = []
    seen: set[str] = set()

    for group in decision.get('merge', []):
        clean = [n for n in group if n in valid_names]
        # Check for duplicates
        new_names = [n for n in clean if n not in seen]
        if len(new_names) < 2:
            if new_names:
                print(f'[warn] Skipping merge group {group}: too few valid names after dedup')
            continue
        merge_groups.append(new_names)
        seen.update(new_names)

    for name in decision.get('delete', []):
        if name not in valid_names:
            print(f'[warn] Skipping delete "{name}": not a valid part')
            continue
        if name in seen:
            print(f'[warn] Skipping delete "{name}": already in a merge group')
            continue
        delete_list.append(name)
        seen.add(name)

    return merge_groups, delete_list


def apply_operations(
    parts: dict[str, trimesh.Trimesh],
    merge_groups: list[list[str]],
    delete_list: list[str],
) -> dict[str, trimesh.Trimesh]:
    """Apply merge and delete operations, returning new parts dict."""
    result: dict[str, trimesh.Trimesh] = {}

    # Track which names are consumed by merge or delete
    consumed: set[str] = set()
    for group in merge_groups:
        consumed.update(group)
    consumed.update(delete_list)

    # Apply merges
    for group in merge_groups:
        merged_name = sorted(group)[0]
        meshes = [parts[n] for n in group]
        merged = trimesh.util.concatenate(meshes)
        result[merged_name] = merged
        print(f'[merge] {" + ".join(group)} -> {merged_name}')

    # Keep untouched parts
    for name, mesh in parts.items():
        if name not in consumed:
            result[name] = mesh

    # Log deletes
    for name in delete_list:
        print(f'[delete] {name}')

    return result


def main(
    parts: Path,
    output: Path = None,
    category: str = '',
    model: str = 'claude-opus-4-6',
    no_render: bool = False,
    render_dir: Path = None,
) -> None:
    """
    Auto-merge over-segmented parts using VLM analysis.

    Args:
        parts: Path to directory of .ply files.
        output: Output directory for merged parts. Default: {parts}/../parts_merged
        category: Object category hint for the VLM (e.g., "faucet", "fridge").
        model: Anthropic model to use.
        no_render: Skip rendering (use existing renders).
        render_dir: Custom render output directory.
    """
    assert parts.exists() and parts.is_dir(), f'{parts} must be an existing directory of .ply files'

    if output is None:
        output = parts.parent / 'parts_merged'
    if render_dir is None:
        render_dir = parts.parent / 'auto_merge_renders'

    # Load part names
    ply_files = sorted(parts.glob('*.ply'))
    assert ply_files, f'No .ply files found in {parts}'
    part_names = [p.stem for p in ply_files]
    print(f'[auto_merge] {len(part_names)} parts: {", ".join(part_names)}')

    # Step 1: Render
    if not no_render:
        render_parts(parts, render_dir)
    else:
        print('[auto_merge] Skipping rendering (--no-render)')

    # Collect rendered images
    image_paths = [render_dir / name for name in VIEW_NAMES]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Missing rendered views: {[str(p) for p in missing]}')

    # Step 2: Build legend
    legend = build_legend(part_names)
    print(f'\n[legend]\n{legend}\n')

    # Step 3: Build prompt
    category_line = f'The object is a **{category}**.' if category else ''
    prompt_text = USER_PROMPT.format(category_line=category_line, legend=legend)

    # Step 4: Call VLM
    print(f'[vlm] Calling {model}...')
    raw_response = call_vlm(image_paths, prompt_text, model)
    print(f'[vlm] Response:\n{raw_response}\n')

    # Step 5: Parse and validate
    try:
        decision = parse_vlm_json(raw_response)
    except json.JSONDecodeError as e:
        print(f'[error] Failed to parse VLM response as JSON: {e}')
        sys.exit(1)

    valid_names = set(part_names)
    merge_groups, delete_list = validate_response(decision, valid_names)

    if not merge_groups and not delete_list:
        print('[auto_merge] No merges or deletes needed.')

    # Step 6: Load meshes and apply operations
    parts_meshes = {p.stem: trimesh.load(p, force='mesh') for p in ply_files}
    final_parts = apply_operations(parts_meshes, merge_groups, delete_list)

    # Step 7: Save output
    output.mkdir(parents=True, exist_ok=True)
    for name, mesh in sorted(final_parts.items()):
        out_path = output / f'{name}.ply'
        mesh.export(str(out_path))

    # Save decision log
    log = {
        'source': str(parts),
        'model': model,
        'category': category,
        'original_parts': part_names,
        'vlm_response': raw_response,
        'merge_groups': merge_groups,
        'deleted': delete_list,
        'final_parts': sorted(final_parts.keys()),
        'legend': legend,
        'parts_description': decision.get('parts_description', {}),
        'reasoning': decision.get('reasoning', ''),
    }
    log_path = output / 'auto_merge.json'
    log_path.write_text(json.dumps(log, indent=2))

    print(f'\n=== auto_merge summary ===')
    print(f'  Input parts  : {len(part_names)}')
    print(f'  Merged groups: {len(merge_groups)}')
    print(f'  Deleted      : {len(delete_list)}')
    print(f'  Final parts  : {len(final_parts)}')
    print(f'  Output dir   : {output}')
    print(f'  Decision log : {log_path}')
    print(f'==========================\n')


if __name__ == '__main__':
    tyro.cli(main)
