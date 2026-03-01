from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List

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

    @staticmethod
    def process(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        total_face = mesh.faces.shape[0]
        return [m for m in mesh.split(only_watertight=False) if m.faces.shape[0] > total_face * 0.02]

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
        result = trimesh.Scene()
        for m in chain.from_iterable(self.process(m) for m in obj_mesh.geometry.values()):
            result.add_geometry(m)
        return result

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
