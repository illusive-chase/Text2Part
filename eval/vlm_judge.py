"""
Use a VLM (via OpenAI-compatible API) to judge 3D part decompositions.

Two criteria per view:
  1. Object quality  — does the shape look like a reasonable {category}?
  2. Fragment detection — are there floating disconnected pieces?

Usage:
    python eval/vlm_judge.py \
        --render_dir /tmp/render_test/faucet0 \
        --category faucet
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

# Ensure localhost traffic bypasses any HTTP proxy.
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

from openai import OpenAI

# ── view file names produced by render_parts.py (8 azimuths, 45° apart) ──
VIEW_NAMES = [f"view_{az:03d}.png" for az in range(0, 360, 45)]

SYSTEM_PROMPT = (
    "You are an expert quality inspector for 3D part decompositions. "
    "Each image shows a 3D object that has been decomposed into colored parts."
)

USER_PROMPT = """\
Look at this rendered view of a 3D **{category}** that was decomposed into colored parts.

Judge the image on TWO independent criteria:

**Criterion 1 — Object quality**
Does the overall 3D shape look like a reasonable {category}? Consider whether the \
proportions, silhouette, and part layout are plausible. Minor cosmetic roughness is \
acceptable; flag only clear problems such as severely distorted proportions, missing \
major components, or a shape that is unrecognizable as a {category}.

**Criterion 2 — Floating fragments**
Are there small, disconnected pieces that appear to hover in space and are clearly \
NOT attached to the main body? Typical examples: stray slivers, tiny orphan clusters, \
or debris floating away from the object.
Things that are NOT floating fragments (do NOT flag these):
- Thin but connected parts like handles, knobs, hinges, or tubes
- Parts that touch or overlap another part even slightly
- Separate functional sub-components (e.g. scissor blades, faucet handles)

Respond with EXACTLY this JSON (no other text):
{{"object_pass": true/false, "object_reasoning": "<one sentence>", \
"fragment_pass": true/false, "fragment_reasoning": "<one sentence>"}}

Set "object_pass" to true if the shape is a reasonable {category}.
Set "fragment_pass" to true if there are NO floating fragments."""


def encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def detect_model(client: OpenAI) -> str:
    models = client.models.list()
    return models.data[0].id


def parse_vlm_json(raw: str) -> dict:
    """Extract JSON from VLM response, tolerant of markdown fences."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def judge_view(
    client: OpenAI,
    model: str,
    image_path: Path,
    category: str,
) -> dict:
    b64 = encode_image_b64(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(category=category)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                        },
                    },
                ],
            },
        ],
        max_tokens=256,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()

    try:
        result = parse_vlm_json(raw)
    except json.JSONDecodeError:
        return {
            "object_pass": False,
            "object_reasoning": f"Unparseable VLM response: {raw}",
            "fragment_pass": False,
            "fragment_reasoning": f"Unparseable VLM response: {raw}",
        }

    return {
        "object_pass": bool(result.get("object_pass", False)),
        "object_reasoning": str(result.get("object_reasoning", "")),
        "fragment_pass": bool(result.get("fragment_pass", False)),
        "fragment_reasoning": str(result.get("fragment_reasoning", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM judge for part decompositions.")
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    model = detect_model(client)
    print(f"Model: {model}")

    views: dict[str, dict] = {}
    for name in VIEW_NAMES:
        img_path = args.render_dir / name
        if not img_path.exists():
            print(f"  [skip] {name} not found")
            continue
        result = judge_view(client, model, img_path, args.category)
        views[name] = result
        obj = "PASS" if result["object_pass"] else "FAIL"
        frag = "PASS" if result["fragment_pass"] else "FAIL"
        print(f"  {name}: object={obj} fragment={frag}")
        if not result["object_pass"]:
            print(f"           object : {result['object_reasoning']}")
        if not result["fragment_pass"]:
            print(f"           fragment: {result['fragment_reasoning']}")

    total = len(views)
    object_score = sum(1 for v in views.values() if v["object_pass"])
    fragment_score = sum(1 for v in views.values() if v["fragment_pass"])

    output = {
        "category": args.category,
        "views": views,
        "object_score": object_score,
        "fragment_score": fragment_score,
    }
    out_path = args.render_dir / "judge.json"
    out_path.write_text(json.dumps(output, indent=2))

    # ── summary ──
    print(f"\n{'=' * 50}")
    print(f"Category       : {args.category}")
    print(f"Object score   : {object_score}/{total} views passed")
    print(f"Fragment score : {fragment_score}/{total} views passed")
    if object_score < total:
        print("Object failures:")
        for name, v in views.items():
            if not v["object_pass"]:
                print(f"  {name}: {v['object_reasoning']}")
    if fragment_score < total:
        print("Fragment failures:")
        for name, v in views.items():
            if not v["fragment_pass"]:
                print(f"  {name}: {v['fragment_reasoning']}")
    print(f"Results        : {out_path}")


if __name__ == "__main__":
    main()
