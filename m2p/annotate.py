from __future__ import annotations

import json
import xml.etree.ElementTree as ET
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
        # currently selected part names
        self.selected: set[str] = set()
        # merge history for undo: list of (merged_name, {original_name: mesh, ...}, {original_name: color_idx, ...})
        self.merge_history: list[tuple[str, dict[str, trimesh.Trimesh], dict[str, int]]] = []
        # next color index
        self._color_idx = 0

        # ========== hinge annotation ==========
        self.hinges: list[dict] = []
        # preview state
        self._preview_idx: int | None = None
        self._preview_slider: Any = None
        self._preview_axis_handle: Any = None

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
        name = event.target.name
        if name not in self.parts:
            return
        if name in self.selected:
            self.selected.discard(name)
        else:
            self.selected.add(name)
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

    def detect_hinge_axis(self, base_name: str, child_name: str) -> dict:
        """Detect hinge axis between two adjacent parts using PCA on boundary vertices."""
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

        # Step 2: PCA for axis direction
        centroid = boundary_points.mean(axis=0)
        centered = boundary_points - centroid
        cov = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        axis_direction = eigenvectors[:, -1].copy()
        axis_direction /= np.linalg.norm(axis_direction)

        # Step 3: consistent direction (align with Y+)
        if np.dot(axis_direction, np.array([0.0, 1.0, 0.0])) < 0:
            axis_direction = -axis_direction

        # Step 4: quality score
        score = eigenvalues[-1] / max(eigenvalues[-2], 1e-12)

        if score < 1.5:
            raise ValueError(f"Cannot detect valid hinge axis (score={score:.2f}).")

        return {'axis': axis_direction, 'pivot': centroid, 'score': score}

    def add_hinge(self, base_name: str, child_name: str) -> str | None:
        """Detect hinge axis and store it."""
        info = self.detect_hinge_axis(base_name, child_name)
        hinge = {
            'base': base_name,
            'child': child_name,
            'axis': info['axis'],
            'pivot': info['pivot'],
            'limits': (-np.pi, np.pi),
            'score': info['score'],
        }
        self.hinges.append(hinge)
        return f"Hinge {len(self.hinges)-1}: {base_name} -> {child_name} (score={info['score']:.2f})"

    def remove_last_hinge(self) -> str | None:
        if not self.hinges:
            return None
        h = self.hinges.pop()
        return f"Removed hinge: {h['base']} -> {h['child']}"

    def start_hinge_preview(self, idx: int) -> None:
        """Enter preview mode for hinge at given index."""
        hinge = self.hinges[idx]
        child_name = hinge['child']
        self._preview_idx = idx

        # Create slider
        self._preview_slider = self.server.gui.add_slider(
            label=f"Hinge {idx} angle",
            min=-np.pi,
            max=np.pi,
            step=0.01,
            initial_value=0.0,
        )

        # Show axis line in scene
        pivot = hinge['pivot']
        axis = hinge['axis']
        positions = np.array([pivot - axis * 0.3, pivot + axis * 0.3])
        self._preview_axis_handle = self.server.scene.add_spline_catmull_rom(
            name="_hinge_axis",
            positions=positions,
            color=(255, 0, 0),
        )

        # Register callback
        self._preview_slider.on_update(
            lambda _: self._apply_hinge_angle(self._preview_slider.value)
        )

    def stop_hinge_preview(self) -> None:
        """Exit preview mode, restore original mesh."""
        if self._preview_idx is None:
            return

        # Remove slider
        if self._preview_slider is not None:
            self._preview_slider.remove()
            self._preview_slider = None

        # Remove axis line
        if self._preview_axis_handle is not None:
            self._preview_axis_handle.remove()
            self._preview_axis_handle = None

        # Restore child handle transform to identity
        child_name = self.hinges[self._preview_idx]['child']
        if child_name in self.handles:
            self.handles[child_name].wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            self.handles[child_name].position = np.array([0.0, 0.0, 0.0])

        self._preview_idx = None

    def _apply_hinge_angle(self, angle: float) -> None:
        """Apply rotation to child mesh for preview via scene node transform."""
        if self._preview_idx is None:
            return

        hinge = self.hinges[self._preview_idx]
        child_name = hinge['child']
        if child_name not in self.handles:
            return

        pivot = hinge['pivot']
        axis = hinge['axis']

        # Quaternion for rotation about axis (w, x, y, z)
        wxyz = trimesh.transformations.quaternion_about_axis(angle, axis)
        # Position offset so that the rotation is about pivot, not origin:
        #   world = R @ local + position  =>  position = pivot - R @ pivot
        R = trimesh.transformations.quaternion_matrix(wxyz)[:3, :3]
        position = pivot - R @ pivot

        handle = self.handles[child_name]
        handle.wxyz = wxyz
        handle.position = position

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
            base_name = hinge['base']
            child_name = hinge['child']
            pivot = hinge['pivot']
            axis = hinge['axis']

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

            # Revolute joint from base to abstract pivot
            px, py, pz = pivot
            ax, ay, az = axis
            revolute = ET.SubElement(robot, 'joint',
                                     name=f'hinge_{i}', type='revolute')
            ET.SubElement(revolute, 'parent', link=base_link_name)
            ET.SubElement(revolute, 'child', link=abstract_name)
            ET.SubElement(revolute, 'origin', xyz=f'{px} {py} {pz}', rpy='0 0 0')
            ET.SubElement(revolute, 'axis', xyz=f'{ax} {ay} {az}')
            ET.SubElement(revolute, 'limit',
                          lower=str(hinge['limits'][0]),
                          upper=str(hinge['limits'][1]),
                          effort='2000.0', velocity='2.0')

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
                    'base': h['base'],
                    'child': h['child'],
                    'axis': h['axis'].tolist(),
                    'pivot': h['pivot'].tolist(),
                    'limits': list(h['limits']),
                }
                for h in self.hinges
            ],
        }
        json_path = output_dir / 'annotation.json'
        json_path.write_text(json.dumps(meta, indent=2))

        if self.hinges:
            self.generate_urdf()

        return glb_path

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

        with server.gui.add_folder('Hinge'):
            hinge_text = server.gui.add_text('Hinges', '0', disabled=True)

            def _update_hinge_text() -> None:
                lines = [str(len(state.hinges))]
                for i, h in enumerate(state.hinges):
                    lines.append(f"  {i}: {h['base']} -> {h['child']}")
                hinge_text.value = '\n'.join(lines)

            add_hinge_btn = server.gui.add_button('Add Hinge')

            @add_hinge_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                if len(state.selected) != 2:
                    event.client.add_notification(
                        title='Error', body='Select exactly 2 parts.', loading=False)
                    return
                names = sorted(state.selected)
                # Larger mesh = base, smaller = child
                if len(state.parts[names[0]].vertices) >= len(state.parts[names[1]].vertices):
                    base_name, child_name = names[0], names[1]
                else:
                    base_name, child_name = names[1], names[0]
                try:
                    result = state.add_hinge(base_name, child_name)
                    event.client.add_notification(title='Hinge Added', body=result, loading=False)
                    _update_hinge_text()
                    state.clear_selection()
                except ValueError as e:
                    event.client.add_notification(title='Detection Failed', body=str(e), loading=False)

            remove_hinge_btn = server.gui.add_button('Remove Last Hinge')

            @remove_hinge_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                result = state.remove_last_hinge()
                if result:
                    event.client.add_notification(title='Removed', body=result, loading=False)
                    _update_hinge_text()
                else:
                    event.client.add_notification(title='Nothing to remove', body='No hinges.', loading=False)

            preview_btn = server.gui.add_button('Preview Hinge')

            @preview_btn.on_click
            def _(event: viser.GuiEvent) -> None:
                if not state.hinges:
                    event.client.add_notification(
                        title='Error', body='No hinges to preview.', loading=False)
                    return
                if state._preview_idx is not None:
                    state.stop_hinge_preview()
                state.start_hinge_preview(len(state.hinges) - 1)

            stop_preview_btn = server.gui.add_button('Stop Preview')

            @stop_preview_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                state.stop_hinge_preview()

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
