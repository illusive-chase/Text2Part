from __future__ import annotations

from pathlib import Path

import trimesh
import tyro
import viser
from matplotlib import colormaps

if __name__ == '__main__':

    def main(
        parts: Path,
        port: int = 6789,
    ) -> None:
        assert parts.exists()
        if parts.is_dir():
            scene = trimesh.Scene()
            for p in parts.glob('*.ply'):
                scene.add_geometry(trimesh.load(p))
        else:
            assert parts.suffix == '.glb'
            scene = trimesh.load_scene(parts)

        colormap = list(colormaps.get_cmap('tab20').colors)

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction('+y')
        for i, m in enumerate(scene.geometry.values()):
            if isinstance(m, trimesh.Trimesh):
                server.scene.add_mesh_simple(
                    vertices=m.vertices,
                    faces=m.faces,
                    name=f'part_{i:02d}',
                    color=colormap[i],
                )

        server.sleep_forever()

    tyro.cli(main)
