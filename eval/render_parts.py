"""
Blender headless script to render multi-view images of part-decomposed 3D objects.

Usage:
    blender --background --python eval/render_parts.py -- \
        --parts_dir outputs/gen4/faucet0/parts \
        --output_dir /tmp/render_test/faucet0

Or via plain Python (Blender's bundled python):
    python eval/render_parts.py \
        --parts_dir outputs/gen4/faucet0/parts \
        --output_dir /tmp/render_test/faucet0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # When run via `blender --background --python script.py -- <args>`,
    # everything after the bare `--` goes to sys.argv for the script.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Render part-decomposed 3D objects.")
    parser.add_argument(
        "--parts_dir",
        type=Path,
        required=True,
        help="Directory containing .ply files (one per part).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where rendered PNGs will be saved.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        default=False,
        help="Skip the Y-up → Z-up rotation (-90° around X). Use if PLY files are already Z-up.",
    )
    return parser.parse_args(argv)


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


def main() -> None:
    import bpy
    import mathutils
    import trimesh

    args = parse_args()
    parts_dir: Path = args.parts_dir
    output_dir: Path = args.output_dir
    no_rotate: bool = args.no_rotate

    assert parts_dir.exists() and parts_dir.is_dir(), f"parts_dir not found: {parts_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(parts_dir.glob("*.ply"))
    assert ply_files, f"No .ply files found in {parts_dir}"

    # ------------------------------------------------------------------
    # 1. Reset scene
    # ------------------------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene

    # ------------------------------------------------------------------
    # 2. Load each .ply as a separate mesh object and assign a vivid color
    # ------------------------------------------------------------------
    n_parts = len(ply_files)
    part_objects: list[bpy.types.Object] = []

    for idx, ply_path in enumerate(ply_files):
        # pip-installed headless bpy doesn't ship the PLY operator;
        # load with trimesh and push vertices+faces directly into bpy.
        mesh_tm = trimesh.load(str(ply_path), force="mesh")
        vertices = [tuple(v) for v in mesh_tm.vertices]
        faces = [tuple(int(i) for i in f) for f in mesh_tm.faces]

        mesh = bpy.data.meshes.new(f"mesh_part_{idx:03d}")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(f"part_{idx:03d}", mesh)
        scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Vivid HSV color with evenly-spaced hues
        hue = idx / n_parts
        r, g, b = hsv_to_rgb(hue, 0.9, 0.9)

        mat = bpy.data.materials.new(name=f"mat_part_{idx:03d}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Specular IOR Level"].default_value = 0.3

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        part_objects.append(obj)

    # ------------------------------------------------------------------
    # 2b. Orientation normalization: Y-up → Z-up
    # ------------------------------------------------------------------
    # PLY files from gen4 are exported with Y-up convention, but Blender
    # uses Z-up. Rotate all parts -90° around the X axis to align them.
    if not no_rotate:
        rot_x_neg90 = mathutils.Matrix.Rotation(math.radians(90), 4, "X")
        for obj in part_objects:
            obj.matrix_world = rot_x_neg90 @ obj.matrix_world

    # ------------------------------------------------------------------
    # 3. Compute combined bounding box
    # ------------------------------------------------------------------
    all_corners: list[mathutils.Vector] = []
    for obj in part_objects:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ mathutils.Vector(corner))

    bbox_min = mathutils.Vector((
        min(v.x for v in all_corners),
        min(v.y for v in all_corners),
        min(v.z for v in all_corners),
    ))
    bbox_max = mathutils.Vector((
        max(v.x for v in all_corners),
        max(v.y for v in all_corners),
        max(v.z for v in all_corners),
    ))
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min
    bounding_sphere_radius = bbox_size.length / 2.0
    camera_distance = 2.5 * bounding_sphere_radius

    # ------------------------------------------------------------------
    # 4. Scene / render settings
    # ------------------------------------------------------------------
    # EEVEE requires EGL (GPU) which isn't available on headless servers;
    # Cycles CPU works without any display or GPU libraries.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64  # fast enough for evaluation renders
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # White background
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0

    # ------------------------------------------------------------------
    # 5. Camera
    # ------------------------------------------------------------------
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # ------------------------------------------------------------------
    # 6. 3-point lighting (key + fill + rim)
    # ------------------------------------------------------------------
    def add_sun(name: str, energy: float, direction: tuple[float, float, float]) -> None:
        light_data = bpy.data.lights.new(name=name, type="SUN")
        light_data.energy = energy
        light_obj = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light_obj)
        # Point the sun by rotating from default -Z direction
        dx, dy, dz = direction
        light_obj.rotation_euler = mathutils.Vector((dx, dy, dz)).to_track_quat(
            "-Z", "Y"
        ).to_euler()

    # Key light: front-right-above
    add_sun("KeyLight",  energy=3.0, direction=(-1.0, -1.0, -1.5))
    # Fill light: front-left, softer
    add_sun("FillLight", energy=1.2, direction=( 1.0, -1.0, -1.0))
    # Rim light: behind-above
    add_sun("RimLight",  energy=1.5, direction=( 0.0,  1.5, -1.0))

    # ------------------------------------------------------------------
    # 7. Render 8 views
    # ------------------------------------------------------------------
    elevation_deg = 20.0
    elevation_rad = math.radians(elevation_deg)
    image_files: list[str] = []

    for view_idx in range(8):
        azimuth_deg = view_idx * 45
        azimuth_rad = math.radians(azimuth_deg)

        # Camera position in spherical coordinates around bbox_center
        cx = bbox_center.x + camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        cy = bbox_center.y + camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        cz = bbox_center.z + camera_distance * math.sin(elevation_rad)

        cam_obj.location = mathutils.Vector((cx, cy, cz))

        # Point camera at bbox_center
        direction = bbox_center - cam_obj.location
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        output_path = output_dir / f"view_{azimuth_deg:03d}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        image_files.append(output_path.name)

    # ------------------------------------------------------------------
    # 8. Summary and verify.json
    # ------------------------------------------------------------------
    part_file_names = [p.name for p in ply_files]
    verify = {
        "num_parts": n_parts,
        "num_images": 8,
        "part_files": part_file_names,
        "image_files": image_files,
    }
    verify_path = output_dir / "verify.json"
    verify_path.write_text(json.dumps(verify, indent=2))

    print(f"\n=== render_parts summary ===")
    print(f"  Parts loaded : {n_parts}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Images saved : {len(image_files)}")
    print(f"  verify.json  : {verify_path}")
    print(f"============================\n")


if __name__ == "__main__":
    main()
