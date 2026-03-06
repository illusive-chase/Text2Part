from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import tyro
import viser
from matplotlib import colormaps

HIGHLIGHT_COLOR = np.array([255, 255, 100, 255], dtype=np.uint8)


class AnnotationState:

    def __init__(self, parts_dir: Path, server: viser.ViserServer) -> None:
        self.parts_dir = parts_dir
        self.server = server
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

        self._load_parts()

    def _load_parts(self) -> None:
        for p in sorted(self.parts_dir.glob('*.ply')):
            mesh = trimesh.load(p, force='mesh')
            name = p.stem
            self.parts[name] = mesh
            self.part_colors[name] = self._color_idx
            self._color_idx = (self._color_idx + 1) % len(self.colormap)

    def _add_mesh(self, name: str, mesh: trimesh.Trimesh, selected: bool = False) -> None:
        vis_mesh = mesh.copy()
        if selected:
            vis_mesh.visual = trimesh.visual.ColorVisuals()
            vis_mesh.visual.face_colors = HIGHLIGHT_COLOR
        else:
            vis_mesh.visual = trimesh.visual.ColorVisuals()
            color = np.append(self.colormap[self.part_colors[name]], 255)
            vis_mesh.visual.face_colors = color

        handle = self.server.scene.add_mesh_trimesh(
            name=name,
            mesh=vis_mesh,
            cast_shadow=False,
            receive_shadow=False,
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
            self.handles[name].remove()
            del self.handles[name]
        if name in self.parts:
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

        output_dir = self.parts_dir.parent
        glb_path = output_dir / 'annotated.glb'
        scene.export(str(glb_path))

        meta = {
            'source': str(self.parts_dir),
            'num_parts': len(self.parts),
            'parts': sorted(self.parts.keys()),
            'merge_history': [
                {
                    'merged_name': name,
                    'original_parts': sorted(orig.keys()),
                }
                for name, orig, _ in self.merge_history
            ],
        }
        json_path = output_dir / 'annotation.json'
        json_path.write_text(json.dumps(meta, indent=2))

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
    ) -> None:
        assert parts.exists() and parts.is_dir(), f'{parts} must be a directory of .ply files'

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction('+y')

        state = AnnotationState(parts, server)

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
