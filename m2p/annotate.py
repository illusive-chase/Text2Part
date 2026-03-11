from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
import tyro
import viser
from matplotlib import colormaps
from scipy.spatial import cKDTree

HIGHLIGHT_COLOR = np.array([255, 255, 100], dtype=np.uint8)


def _direction_to_wxyz(direction: np.ndarray) -> np.ndarray:
    """Convert a direction vector to a wxyz quaternion (rotate +Z to align with direction)."""
    z = np.array([0, 0, 1.0])
    d = direction / np.linalg.norm(direction)
    cross = np.cross(z, d)
    dot = np.dot(z, d)
    if np.linalg.norm(cross) < 1e-6:
        return np.array([1, 0, 0, 0] if dot > 0 else [0, 0, 1, 0], dtype=np.float32)
    half = np.arctan2(np.linalg.norm(cross), dot) / 2
    ax = cross / np.linalg.norm(cross)
    return np.array([np.cos(half), *(ax * np.sin(half))], dtype=np.float32)


ARROW_COLOR = np.array([255, 50, 50], dtype=np.uint8)
TRANSLATION_ARROW_COLOR = np.array([50, 150, 255], dtype=np.uint8)


@lru_cache()
def _make_canonical_arrow(
    length: float,
    shaft_radius: float = 0.005,
    head_angle: float = 30,
    head_fraction: float = 0.07,
) -> trimesh.Trimesh:
    """Arrow along +Z from origin, *length* units tall."""
    shaft_len = length * (1 - head_fraction)
    head_len = length * head_fraction
    head_radius = np.tan(np.deg2rad(head_angle / 2)) * head_len
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_len, sections=12)
    shaft.apply_translation([0, 0, shaft_len / 2])
    head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=12)
    head.apply_translation([0, 0, shaft_len])
    return trimesh.util.concatenate([shaft, head])


@dataclass
class HingeState:
    base_name: str
    child_name: str
    ep1: np.ndarray  # endpoint 1 position (3,)
    ep2: np.ndarray  # endpoint 2 position (3,)
    score: float  # PCA quality score
    original_mesh: trimesh.Trimesh  # child mesh snapshot at angle=0
    arrow_length: float = 0.0  # canonical arrow length (for scale computation)
    limit_min: float = -np.pi  # user-set angular limit (min)
    limit_max: float = np.pi  # user-set angular limit (max)
    # scene/GUI handles (set after creation)
    slider: viser.GuiSliderHandle = None  # viser GUI slider
    axes_handle: viser.MeshHandle = None  # arrow mesh handle (ep1 → ep2)
    ctrl_ep1: viser.TransformControlsHandle = None  # TransformControlsHandle at ep1
    ctrl_ep2: viser.TransformControlsHandle = None  # TransformControlsHandle at ep2
    btn_group: viser.GuiButtonGroupHandle = None
    joint_type: str = 'revolute'  # 'revolute' or 'prismatic'
    _default_limit_min: float = -np.pi  # initial slider min (for reset)
    _default_limit_max: float = np.pi  # initial slider max (for reset)

    @property
    def axis(self) -> np.ndarray:
        d = self.ep2 - self.ep1
        return d / np.linalg.norm(d)

    @property
    def pivot(self) -> np.ndarray:
        return (self.ep1 + self.ep2) / 2.0


class AnnotationState:

    def __init__(self, parts_path: Path, server: viser.ViserServer, max_faces: int = 10000) -> None:
        self.parts_path = parts_path
        self.server = server
        self.max_faces = max_faces
        self.colormap = [
            (np.array(c[:3]) * 255).astype(np.uint8)
            for c in colormaps.get_cmap('tab20').colors
        ]

        # name -> mesh
        self.parts: dict[str, trimesh.Trimesh] = {}
        # name -> color index in colormap
        self.part_colors: dict[str, int] = {}
        # scene handles
        self.handles: dict[str, Any] = {}
        # currently selected part names (ordered by click)
        self.selected: list[str] = []
        # merge history for undo: list of (merged_name, {original_name: mesh, ...}, {original_name: color_idx, ...})
        self.merge_history: list[tuple[str, dict[str, trimesh.Trimesh], dict[str, int]]] = []
        # delete history for undo: list of (part_name, mesh, color_index)
        self.delete_history: list[tuple[str, trimesh.Trimesh, int]] = []
        # next color index
        self._color_idx = 0

        # ========== hinge annotation ==========
        self.hinges: list[HingeState] = []

        self._load_parts()

    def _load_parts(self) -> None:
        if self.parts_path.is_dir():
            self._load_parts_from_dir()
        else:
            self._load_parts_from_glb()

    def _simplify_mesh(self, mesh: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
        if self.max_faces <= 0 or len(mesh.faces) <= self.max_faces:
            return mesh
        orig = len(mesh.faces)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=self.max_faces)
        simplified = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
        )
        print(f"[simplify] {name}: {orig} -> {len(simplified.faces)} faces")
        return simplified

    def _load_parts_from_dir(self) -> None:
        for p in sorted(self.parts_path.glob('*.ply')):
            mesh = trimesh.load(p, force='mesh')
            name = p.stem
            mesh = self._simplify_mesh(mesh, name)
            self.parts[name] = mesh
            self.part_colors[name] = self._color_idx
            self._color_idx = (self._color_idx + 1) % len(self.colormap)

    def _load_parts_from_glb(self) -> None:
        scene = trimesh.load(str(self.parts_path))
        if isinstance(scene, trimesh.Trimesh):
            # Single mesh, no scene graph
            scene = self._simplify_mesh(scene, '00')
            self.parts['00'] = scene
            self.part_colors['00'] = self._color_idx
            self._color_idx = (self._color_idx + 1) % len(self.colormap)
            return
        for name in sorted(scene.graph.nodes_geometry):
            transform, geom_name = scene.graph[name]
            mesh = scene.geometry[geom_name].copy()
            mesh.apply_transform(transform)
            mesh = self._simplify_mesh(mesh, name)
            self.parts[name] = mesh
            self.part_colors[name] = self._color_idx
            self._color_idx = (self._color_idx + 1) % len(self.colormap)

    @property
    def output_dir(self) -> Path:
        if self.parts_path.is_dir():
            return self.parts_path.parent
        return self.parts_path.parent

    def _add_mesh(self, name: str, mesh: trimesh.Trimesh, selected: bool = False) -> None:
        handle = self.server.scene.add_mesh_simple(
            name=name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=HIGHLIGHT_COLOR if selected else self.colormap[self.part_colors[name]],
        )
        handle.on_click(lambda event: self._handle_click(event))
        self.handles[name] = handle

    def _handle_click(self, event: viser.SceneNodePointerEvent) -> None:
        name = event.target.name.removeprefix('/')
        if name not in self.parts:
            return
        if name in self.selected:
            self.selected.remove(name)
        else:
            self.selected.append(name)
        self._refresh_mesh(name)
        self._update_selection_text()

    def _refresh_mesh(self, name: str) -> None:
        if name in self.handles:
            color = HIGHLIGHT_COLOR if name in self.selected else self.colormap[self.part_colors[name]]
            self.handles[name].color = color
        elif name in self.parts:
            self._add_mesh(name, self.parts[name], selected=name in self.selected)

    def refresh_all(self) -> None:
        for name in list(self.handles):
            self.handles[name].remove()
        self.handles.clear()
        for name, mesh in self.parts.items():
            self._add_mesh(name, mesh, selected=name in self.selected)
        self._update_selection_text()

    def clear_selection(self) -> None:
        prev = list(self.selected)
        self.selected.clear()
        for name in prev:
            self._refresh_mesh(name)
        self._update_selection_text()

    def merge_selected(self) -> str | None:
        if len(self.selected) < 2:
            return None

        selected_names = sorted(self.selected)
        merged = trimesh.util.concatenate([self.parts[n] for n in selected_names])

        # Determine merged name: use lowest original name
        merged_name = selected_names[0]
        merged_color = self.part_colors[selected_names[0]]

        # Save undo info
        original_parts = {n: self.parts[n] for n in selected_names}
        original_colors = {n: self.part_colors[n] for n in selected_names}
        self.merge_history.append((merged_name, original_parts, original_colors))

        # Remove old parts
        for name in selected_names:
            if name in self.handles:
                self.handles[name].remove()
                del self.handles[name]
            del self.parts[name]
            del self.part_colors[name]

        # Add merged part
        self.parts[merged_name] = merged
        self.part_colors[merged_name] = merged_color
        self.selected.clear()
        self._add_mesh(merged_name, merged, selected=False)
        self._update_selection_text()

        return f"Merged {', '.join(selected_names)} -> {merged_name}"

    def _detect_boundary_pca(self, base_name: str, child_name: str) -> dict:
        """Shared boundary PCA: KDTree extraction + PCA. Returns eigenvalues/eigenvectors for callers to interpret."""
        base_mesh = self.parts[base_name]
        child_mesh = self.parts[child_name]

        # Step 1: extract boundary vertices
        base_tree = cKDTree(base_mesh.vertices)
        child_tree = cKDTree(child_mesh.vertices)

        # threshold = child mesh average edge length * 1.5
        edges = child_mesh.vertices[child_mesh.edges_unique]
        avg_edge_len = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).mean()
        threshold = avg_edge_len * 1.5

        # child vertices close to base
        dists_c2b, _ = base_tree.query(child_mesh.vertices)
        child_boundary = child_mesh.vertices[dists_c2b < threshold]

        # base vertices close to child
        dists_b2c, _ = child_tree.query(base_mesh.vertices)
        base_boundary = base_mesh.vertices[dists_b2c < threshold]

        if len(child_boundary) + len(base_boundary) < 3:
            raise ValueError("Parts are not adjacent — too few boundary vertices.")

        boundary_points = np.vstack([child_boundary, base_boundary])

        # Step 2: PCA
        centroid = boundary_points.mean(axis=0)
        centered = boundary_points - centroid
        cov = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order: eigenvalues[0] smallest, eigenvalues[-1] largest

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'boundary_centroid': centroid,
            'avg_edge_len': avg_edge_len,
        }

    def detect_hinge_axis(self, base_name: str, child_name: str) -> dict:
        """
        Detect hinge axis between two adjacent parts using PCA on boundary vertices.
        Uses the largest eigenvector (fit a line along the elongated boundary).
        """
        pca = self._detect_boundary_pca(base_name, child_name)
        eigenvalues = pca['eigenvalues']
        eigenvectors = pca['eigenvectors']
        centroid = pca['boundary_centroid']
        avg_edge_len = pca['avg_edge_len']

        # Largest eigenvector = direction the boundary is elongated along
        axis_direction = eigenvectors[:, -1].copy()
        axis_direction /= np.linalg.norm(axis_direction)
        if np.dot(axis_direction, np.array([0.0, 1.0, 0.0])) < 0:
            axis_direction = -axis_direction

        # Quality: ratio of largest to second-largest eigenvalue
        score = eigenvalues[-1] / max(eigenvalues[-2], 1e-12)
        if score < 1.2:
            raise ValueError(f"Cannot detect valid hinge axis (score={score:.2f}).")

        # Compute extent of boundary along axis for endpoint placement
        child_mesh = self.parts[child_name]
        base_mesh = self.parts[base_name]
        boundary_points = np.vstack([
            child_mesh.vertices[cKDTree(base_mesh.vertices).query(child_mesh.vertices)[0] < avg_edge_len * 1.5],
            base_mesh.vertices[cKDTree(child_mesh.vertices).query(base_mesh.vertices)[0] < avg_edge_len * 1.5],
        ])
        centered = boundary_points - centroid
        projections = centered @ axis_direction
        extent = max((projections.max() - projections.min()) / 2, avg_edge_len * 5) * 2

        return {'axis': axis_direction, 'pivot': centroid, 'score': score, 'extent': extent}

    def detect_translation_axis(self, base_name: str, child_name: str) -> dict:
        """
        Detect translation axis between two adjacent parts using PCA on boundary vertices.
        Uses the smallest eigenvector (normal to the contact plane).
        """
        pca = self._detect_boundary_pca(base_name, child_name)
        eigenvalues = pca['eigenvalues']
        eigenvectors = pca['eigenvectors']

        # Smallest eigenvector = normal to the flat contact surface
        axis_direction = eigenvectors[:, 0].copy()
        axis_direction /= np.linalg.norm(axis_direction)
        if np.dot(axis_direction, np.array([0.0, 1.0, 0.0])) < 0:
            axis_direction = -axis_direction

        # Quality: ratio of second-smallest to smallest eigenvalue (flatter = better)
        score = eigenvalues[1] / max(eigenvalues[0], 1e-12)
        if score < 1.2:
            raise ValueError(f"Cannot detect valid translation axis (score={score:.2f}).")

        # Axis position = child bbox center
        child_mesh = self.parts[child_name]
        bbox_center = child_mesh.bounding_box.centroid

        # Extent based on child mesh vertex projections along axis
        child_centered = child_mesh.vertices - bbox_center
        projections = child_centered @ axis_direction
        child_half_extent = (projections.max() - projections.min()) / 2
        extent = child_half_extent * 1.5

        return {
            'axis': axis_direction,
            'pivot': bbox_center,
            'score': score,
            'extent': extent,
            'child_half_extent': child_half_extent,
        }

    def _setup_joint_gui(self, hinge: HingeState, idx: int) -> None:
        """Shared setup: arrow visualization, transform controls, and callbacks."""
        ep1, ep2 = hinge.ep1, hinge.ep2
        is_prismatic = hinge.joint_type == 'prismatic'
        prefix = 'translation' if is_prismatic else 'hinge'
        arrow_color = TRANSLATION_ARROW_COLOR if is_prismatic else ARROW_COLOR

        # Create arrow mesh visualization (canonical along +Z, oriented via wxyz)
        length = np.linalg.norm(ep2 - ep1)
        direction = (ep2 - ep1) / length
        arrow = _make_canonical_arrow(1.0)
        hinge.arrow_length = length
        hinge.axes_handle = self.server.scene.add_mesh_simple(
            name=f'_{prefix}_axis_{idx}',
            vertices=arrow.vertices * np.array([1.0, 1.0, length]),
            faces=arrow.faces,
            color=arrow_color,
            wxyz=_direction_to_wxyz(direction),
            position=tuple(ep1),
        )

        # Create transform controls (initially hidden)
        hinge.ctrl_ep1 = self.server.scene.add_transform_controls(
            name=f'/_{prefix}_ctrl_{idx}_ep1',
            scale=0.15,
            disable_rotations=True,
            position=tuple(ep1),
            visible=False,
        )
        hinge.ctrl_ep2 = self.server.scene.add_transform_controls(
            name=f'/_{prefix}_ctrl_{idx}_ep2',
            scale=0.15,
            disable_rotations=True,
            position=tuple(ep2),
            visible=False,
        )

        # Register slider callback
        def _on_slider_update(_, _idx=idx):
            self._apply_joint(_idx)

        hinge.slider.on_update(_on_slider_update)

        # Register endpoint update callbacks
        def _on_ep1_update(_, _idx=idx):
            h = self.hinges[_idx]
            h.ep1 = np.array(h.ctrl_ep1.position)
            self._update_hinge_visualization(_idx)
            self._apply_joint(_idx)

        def _on_ep2_update(_, _idx=idx):
            h = self.hinges[_idx]
            h.ep2 = np.array(h.ctrl_ep2.position)
            self._update_hinge_visualization(_idx)
            self._apply_joint(_idx)

        hinge.ctrl_ep1.on_update(_on_ep1_update)
        hinge.ctrl_ep2.on_update(_on_ep2_update)

        # Per-joint axis range button group
        label = 'Translation' if is_prismatic else 'Hinge'
        hinge.btn_group = self.server.gui.add_button_group(f'Limits ({label} {idx})', ['Set Min', 'Set Max', 'Reset'])

        def _on_limit_btn(event, _idx=idx):
            h = self.hinges[_idx]
            clicked = event.target.value
            if clicked == 'Set Min':
                h.slider.min = h.slider.value
                h.limit_min = h.slider.value
            elif clicked == 'Set Max':
                h.slider.max = h.slider.value
                h.limit_max = h.slider.value
            elif clicked == 'Reset':
                h.slider.min = h._default_limit_min
                h.slider.max = h._default_limit_max
                h.slider.value = 0.0
                h.limit_min = h._default_limit_min
                h.limit_max = h._default_limit_max
                self._apply_joint(_idx)

        hinge.btn_group.on_click(_on_limit_btn)

    def add_hinge(self, base_name: str, child_name: str) -> str | None:
        """Detect hinge axis, create visualization and slider."""
        info = self.detect_hinge_axis(base_name, child_name)
        idx = len(self.hinges)

        pivot, axis, extent = info['pivot'], info['axis'], info['extent']
        ep1 = pivot - axis * extent
        ep2 = pivot + axis * extent

        hinge = HingeState(
            base_name=base_name,
            child_name=child_name,
            ep1=ep1,
            ep2=ep2,
            score=info['score'],
            original_mesh=self.parts[child_name].copy(),
        )
        hinge.slider = self.server.gui.add_slider(
            label=f'Hinge {idx}',
            min=-np.pi,
            max=np.pi,
            step=0.01,
            initial_value=0.0,
        )
        self._setup_joint_gui(hinge, idx)
        self.hinges.append(hinge)
        return f"Hinge {idx}: {base_name} -> {child_name} (score={info['score']:.2f})"

    def add_translation(self, base_name: str, child_name: str) -> str | None:
        """Detect translation axis, create visualization and slider."""
        info = self.detect_translation_axis(base_name, child_name)
        idx = len(self.hinges)

        bbox_center, axis, extent = info['pivot'], info['axis'], info['extent']
        child_half_extent = info['child_half_extent']
        ep1 = bbox_center - axis * extent
        ep2 = bbox_center + axis * extent
        slider_range = child_half_extent * 2

        hinge = HingeState(
            base_name=base_name,
            child_name=child_name,
            ep1=ep1,
            ep2=ep2,
            score=info['score'],
            original_mesh=self.parts[child_name].copy(),
            joint_type='prismatic',
            limit_min=-slider_range,
            limit_max=slider_range,
            _default_limit_min=-slider_range,
            _default_limit_max=slider_range,
        )
        hinge.slider = self.server.gui.add_slider(
            label=f'Translation {idx}',
            min=-slider_range,
            max=slider_range,
            step=0.001,
            initial_value=0.0,
        )
        self._setup_joint_gui(hinge, idx)
        self.hinges.append(hinge)
        return f"Translation {idx}: {base_name} -> {child_name} (score={info['score']:.2f})"

    def _apply_joint(self, idx: int) -> None:
        """Apply the current slider value to the child mesh (rotation or translation)."""
        hinge = self.hinges[idx]
        handle = self.handles[hinge.child_name]
        if hinge.joint_type == 'prismatic':
            handle.wxyz = np.array([1, 0, 0, 0])
            handle.position = hinge.slider.value * hinge.axis
        else:
            angle = hinge.slider.value
            wxyz = trimesh.transformations.quaternion_about_axis(angle, hinge.axis)
            R = trimesh.transformations.quaternion_matrix(wxyz)[:3, :3]
            handle.wxyz = wxyz
            handle.position = hinge.pivot - R @ hinge.pivot

    def _update_hinge_visualization(self, idx: int) -> None:
        """Update arrow orientation/position/scale when endpoints change."""
        hinge = self.hinges[idx]
        d = hinge.ep2 - hinge.ep1
        length = np.linalg.norm(d)
        if length < 1e-8:
            return
        direction = d / length
        hinge.axes_handle.wxyz = _direction_to_wxyz(direction)
        hinge.axes_handle.position = hinge.ep1
        hinge.axes_handle.vertices = _make_canonical_arrow(1.0).vertices * np.array([1.0, 1.0, length])

    def remove_last_hinge(self) -> str | None:
        if not self.hinges:
            return None
        h = self.hinges.pop()
        # Clean up GUI and scene elements
        h.slider.remove()
        h.axes_handle.remove()
        if h.ctrl_ep1 is not None:
            h.ctrl_ep1.remove()
        h.ctrl_ep2.remove()
        h.btn_group.remove()
        # Restore original child mesh and reset applied transform
        self.parts[h.child_name] = h.original_mesh
        if h.child_name in self.handles:
            self.handles[h.child_name].wxyz = np.array([1, 0, 0, 0], dtype=np.float32)
            self.handles[h.child_name].position = np.array([0, 0, 0], dtype=np.float32)
        self._refresh_mesh(h.child_name)
        joint_label = 'translation' if h.joint_type == 'prismatic' else 'hinge'
        return f"Removed {joint_label}: {h.base_name} -> {h.child_name}"

    def generate_urdf(self) -> Path:
        """Generate a URDF file for all annotated hinges."""
        output_dir = self.output_dir
        tmp_dir = output_dir / '.tmp_urdf'
        tmp_dir.mkdir(exist_ok=True)

        robot = ET.Element('robot', name='annotated_object')

        # World root link
        ET.SubElement(robot, 'link', name='world')

        inertial_defaults = {'mass': '1.0', 'ixx': '0.01', 'iyy': '0.01', 'izz': '0.01'}

        def add_inertial(link_elem: ET.Element) -> None:
            inertial = ET.SubElement(link_elem, 'inertial')
            ET.SubElement(inertial, 'mass', value=inertial_defaults['mass'])
            ET.SubElement(inertial, 'inertia',
                          ixx=inertial_defaults['ixx'], iyy=inertial_defaults['iyy'],
                          izz=inertial_defaults['izz'], ixy='0', ixz='0', iyz='0')

        for i, hinge in enumerate(self.hinges):
            base_name = hinge.base_name
            child_name = hinge.child_name
            pivot = hinge.pivot
            axis = hinge.axis

            # Export meshes to OBJ
            base_obj = tmp_dir / f'{base_name}.obj'
            child_obj = tmp_dir / f'{child_name}.obj'
            self.parts[base_name].export(str(base_obj))
            self.parts[child_name].export(str(child_obj))

            # Base link
            base_link_name = f'base_{i}' if i > 0 else 'base'
            base_link = ET.SubElement(robot, 'link', name=base_link_name)
            visual = ET.SubElement(base_link, 'visual')
            geom = ET.SubElement(visual, 'geometry')
            ET.SubElement(geom, 'mesh', filename=str(base_obj))
            add_inertial(base_link)

            # Fixed joint from world to base
            fixed_world = ET.SubElement(robot, 'joint',
                                        name=f'joint_fixed_world_{i}', type='fixed')
            ET.SubElement(fixed_world, 'parent', link='world')
            ET.SubElement(fixed_world, 'child', link=base_link_name)

            # Abstract pivot link
            abstract_name = f'abstract_pivot_{i}'
            abstract_link = ET.SubElement(robot, 'link', name=abstract_name)
            add_inertial(abstract_link)

            # Joint from base to abstract pivot (revolute or prismatic)
            px, py, pz = pivot
            ax, ay, az = axis
            if hinge.joint_type == 'prismatic':
                joint_name = f'translation_{i}'
                joint_type = 'prismatic'
                effort_val = '500.0'
                velocity_val = '0.5'
            else:
                joint_name = f'hinge_{i}'
                joint_type = 'revolute'
                effort_val = '2000.0'
                velocity_val = '2.0'
            joint_elem = ET.SubElement(robot, 'joint',
                                       name=joint_name, type=joint_type)
            ET.SubElement(joint_elem, 'parent', link=base_link_name)
            ET.SubElement(joint_elem, 'child', link=abstract_name)
            ET.SubElement(joint_elem, 'origin', xyz=f'{px} {py} {pz}', rpy='0 0 0')
            ET.SubElement(joint_elem, 'axis', xyz=f'{ax} {ay} {az}')
            ET.SubElement(joint_elem, 'limit',
                          lower=str(hinge.limit_min),
                          upper=str(hinge.limit_max),
                          effort=effort_val, velocity=velocity_val)

            # Child link
            child_link_name = f'child_{i}'
            child_link = ET.SubElement(robot, 'link', name=child_link_name)
            visual_c = ET.SubElement(child_link, 'visual')
            geom_c = ET.SubElement(visual_c, 'geometry')
            ET.SubElement(geom_c, 'mesh', filename=str(child_obj))
            add_inertial(child_link)

            # Fixed joint from abstract to child (negative pivot offset)
            fixed_child = ET.SubElement(robot, 'joint',
                                        name=f'joint_fixed_child_{i}', type='fixed')
            ET.SubElement(fixed_child, 'parent', link=abstract_name)
            ET.SubElement(fixed_child, 'child', link=child_link_name)
            ET.SubElement(fixed_child, 'origin',
                          xyz=f'{-px} {-py} {-pz}', rpy='0 0 0')

        tree = ET.ElementTree(robot)
        ET.indent(tree, space='  ')
        urdf_path = output_dir / 'articulated.urdf'
        tree.write(str(urdf_path), xml_declaration=True, encoding='unicode')
        return urdf_path

    def undo_merge(self) -> str | None:
        if not self.merge_history:
            return None

        merged_name, original_parts, original_colors = self.merge_history.pop()

        # Remove the merged mesh
        if merged_name in self.handles:
            self.handles[merged_name].remove()
            del self.handles[merged_name]
        if merged_name in self.parts:
            del self.parts[merged_name]
        if merged_name in self.part_colors:
            del self.part_colors[merged_name]

        # Restore original parts
        for name, mesh in original_parts.items():
            self.parts[name] = mesh
            self.part_colors[name] = original_colors[name]
            self._add_mesh(name, mesh, selected=False)

        self.selected.clear()
        self._update_selection_text()
        return f"Restored {', '.join(sorted(original_parts.keys()))}"

    def delete_selected(self) -> str | None:
        if not self.selected:
            return None

        # Block delete if any selected part is referenced by a hinge
        for name in self.selected:
            for h in self.hinges:
                if name == h.base_name or name == h.child_name:
                    return f"ERROR: '{name}' is used by hinge {h.base_name} -> {h.child_name}. Remove the hinge first."

        deleted_names = sorted(self.selected)
        for name in deleted_names:
            self.delete_history.append((name, self.parts[name], self.part_colors[name]))
            if name in self.handles:
                self.handles[name].remove()
                del self.handles[name]
            del self.parts[name]
            del self.part_colors[name]

        self.selected.clear()
        self._update_selection_text()
        return f"Deleted: {', '.join(deleted_names)}"

    def undo_delete(self) -> str | None:
        if not self.delete_history:
            return None

        name, mesh, color_idx = self.delete_history.pop()
        self.parts[name] = mesh
        self.part_colors[name] = color_idx
        self._add_mesh(name, mesh, selected=False)
        self._update_selection_text()
        return f"Restored: {name}"

    def export(self) -> Path:
        scene = trimesh.Scene()
        for name, mesh in sorted(self.parts.items()):
            scene.add_geometry(mesh, node_name=name)

        output_dir = self.output_dir
        glb_path = output_dir / 'annotated.glb'
        scene.export(str(glb_path))

        meta = {
            'source': str(self.parts_path),
            'num_parts': len(self.parts),
            'parts': sorted(self.parts.keys()),
            'merge_history': [
                {
                    'merged_name': name,
                    'original_parts': sorted(orig.keys()),
                }
                for name, orig, _ in self.merge_history
            ],
            'hinges': [
                {
                    'base': h.base_name,
                    'child': h.child_name,
                    'joint_type': h.joint_type,
                    'axis': h.axis.tolist(),
                    'pivot': h.pivot.tolist(),
                    'limits': [h.limit_min, h.limit_max],
                }
                for h in self.hinges
            ],
        }
        json_path = output_dir / 'annotation.json'
        json_path.write_text(json.dumps(meta, indent=2))

        if self.hinges:
            self.generate_urdf()

        return glb_path

    def set_edit_endpoints_visible(self, visible: bool) -> None:
        """Toggle visibility of all endpoint transform controls."""
        for h in self.hinges:
            if h.ctrl_ep1 is not None:
                h.ctrl_ep1.visible = visible
            h.ctrl_ep2.visible = visible

    # GUI text handle, set after GUI creation
    selection_text: Any = None

    def _update_selection_text(self) -> None:
        if self.selection_text is None:
            return
        if self.selected:
            self.selection_text.value = ', '.join(sorted(self.selected))
        else:
            self.selection_text.value = 'None'


if __name__ == '__main__':

    def main(
        parts: Path,
        port: int = 6789,
        max_faces: int = 10000,
    ) -> None:
        """
        Annotate parts with merge and hinge tools.

        Args:
            parts: Path to a directory of .ply files or a .glb/.gltf file.
            port: Viser server port.
            max_faces: Max faces per part mesh after simplification. Set 0 to disable.
        """
        assert parts.exists(), f'{parts} does not exist'
        assert parts.is_dir() or parts.suffix in ('.glb', '.gltf'), \
            f'{parts} must be a directory of .ply files or a .glb/.gltf file'

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction('+y')

        state = AnnotationState(parts, server, max_faces=max_faces)

        # --- GUI ---
        with server.gui.add_folder('Selection'):
            state.selection_text = server.gui.add_text('Selected', 'None', disabled=True)

            clear_btn = server.gui.add_button('Clear Selection')

            @clear_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                state.clear_selection()

        with server.gui.add_folder('Actions'):
            merge_btn = server.gui.add_button('Merge Selected')

            @merge_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.merge_selected()
                if result:
                    event.client.add_notification(title='Merged', body=result, loading=False)
                else:
                    event.client.add_notification(title='Nothing to merge', body='Select 2+ parts first.', loading=False)

            undo_btn = server.gui.add_button('Undo Last Merge')

            @undo_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.undo_merge()
                if result:
                    event.client.add_notification(title='Undone', body=result, loading=False)
                else:
                    event.client.add_notification(title='Nothing to undo', body='No merge history.', loading=False)

            delete_btn = server.gui.add_button('Delete Selected')

            @delete_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.delete_selected()
                if result is None:
                    event.client.add_notification(title='Nothing to delete', body='Select 1+ parts first.', loading=False)
                elif result.startswith('ERROR:'):
                    event.client.add_notification(title='Cannot delete', body=result[7:], loading=False)
                else:
                    event.client.add_notification(title='Deleted', body=result, loading=False)

            undo_delete_btn = server.gui.add_button('Undo Last Delete')

            @undo_delete_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.undo_delete()
                if result:
                    event.client.add_notification(title='Restored', body=result, loading=False)
                else:
                    event.client.add_notification(title='Nothing to undo', body='No delete history.', loading=False)

        with server.gui.add_folder('Joints'):
            hinge_text = server.gui.add_text('Joints', '0', disabled=True)

            def _update_hinge_text() -> None:
                lines = [str(len(state.hinges))]
                for i, h in enumerate(state.hinges):
                    prefix = '[T]' if h.joint_type == 'prismatic' else '[R]'
                    lines.append(f'  {prefix} {i}: {h.base_name} -> {h.child_name} (score={h.score:.2f})')
                hinge_text.value = '\n'.join(lines)

            edit_cb = server.gui.add_checkbox('Edit Endpoints', initial_value=False)

            @edit_cb.on_update
            def _(_) -> None:
                state.set_edit_endpoints_visible(edit_cb.value)

            def _add_joint(event: viser.GuiEvent, add_fn, label: str) -> None:
                if len(state.selected) != 2:
                    event.client.add_notification(
                        title='Error', body='Select exactly 2 parts.', loading=False)
                    return
                # First selected = child (part), second selected = base
                child_name = state.selected[0]
                base_name = state.selected[1]
                try:
                    result = add_fn(base_name, child_name)
                    event.client.add_notification(title=f'{label} Added', body=result, loading=False)
                    _update_hinge_text()
                    state.clear_selection()
                    if edit_cb.value:
                        h = state.hinges[-1]
                        h.ctrl_ep1.visible = True
                        h.ctrl_ep2.visible = True
                except ValueError as e:
                    event.client.add_notification(title='Detection Failed', body=str(e), loading=False)

            add_hinge_btn = server.gui.add_button('Add Hinge')

            @add_hinge_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                _add_joint(event, state.add_hinge, 'Hinge')

            add_translation_btn = server.gui.add_button('Add Translation')

            @add_translation_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                _add_joint(event, state.add_translation, 'Translation')

            remove_hinge_btn = server.gui.add_button('Remove Last Joint')

            @remove_hinge_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.remove_last_hinge()
                if result:
                    event.client.add_notification(title='Removed', body=result, loading=False)
                    _update_hinge_text()
                    edit_cb.value = False
                    state.set_edit_endpoints_visible(False)
                else:
                    event.client.add_notification(title='Nothing to remove', body='No hinges.', loading=False)

        with server.gui.add_folder('Export'):
            export_btn = server.gui.add_button('Export GLB + JSON')

            @export_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                glb_path = state.export()
                event.client.add_notification(title='Exported', body=str(glb_path), loading=False)

        # Display all parts
        state.refresh_all()

        server.sleep_forever()

    tyro.cli(main)
