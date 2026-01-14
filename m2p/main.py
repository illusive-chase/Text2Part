from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import trimesh
import tyro

sys.path.insert(0, './third_party/hunyuan3dpart/XPart')

from third_party.hunyuan3dpart.XPart.partgen.partformer_pipeline import PartFormerPipeline


@dataclass
class Mesh2Part:

    def init(self, device: torch.device) -> None:
        self.pipe = PartFormerPipeline.from_pretrained(
            model_path="tencent/Hunyuan3D-Part",
            verbose=True,
            device=device,
        )
        self.pipe.to(device=device, dtype=torch.float32)
        self.device = device

    def inference(
        self,
        mesh: Path,
        *,
        seed: int = 42,
    ) -> trimesh.Scene:
        obj_mesh, _ = self.pipe(
            mesh_path=mesh,
            octree_resolution=512,
            output_type='trimesh',
            seed=seed,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        return obj_mesh

if __name__ == '__main__':

    def main(
        mesh: Path,
        output: Path,
        seed: int = 42,
    ) -> None:
        model = Mesh2Part()
        model.init('cuda')
        part = model.inference(mesh, seed=seed)
        if output.suffix == '.glb':
            part.export(output)
        else:
            assert not output.is_file()
            output.mkdir(exist_ok=True, parents=True)
            for i, m in enumerate(part.geometry.values()):
                if isinstance(m, trimesh.Trimesh):
                    m.export(output / f'{i:02d}.ply')

    tyro.cli(main)
