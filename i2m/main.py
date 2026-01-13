from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import trimesh
import tyro
from PIL import Image

sys.path.insert(0, './third_party/hunyuan3d/hy3dshape')

from third_party.hunyuan3d.hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from third_party.hunyuan3d.hy3dshape.hy3dshape.rembg import BackgroundRemover


@dataclass
class Image2Mesh:

    def init(self, device: torch.device) -> None:
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1', device=device)
        self.device = device

    def inference(
        self,
        image: Path,
        *,
        remove_bg: bool = True,
        seed: int = 42,
    ) -> trimesh.Trimesh:
        image = Image.open(image)
        if remove_bg:
            rembg = BackgroundRemover()
            image = rembg(image)
        return self.pipe(
            image=image,
            octree_resolution=384,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )[0]


if __name__ == '__main__':

    def main(
        image: Path,
        output: Path,
        seed: int = 42,
    ) -> None:
        assert output.suffix in ['.obj', '.glb', '.ply']
        model = Image2Mesh()
        model.init('cuda')
        mesh = model.inference(image, seed=seed)
        output.parent.mkdir(exist_ok=True, parents=True)
        mesh.export(output)

    tyro.cli(main)
