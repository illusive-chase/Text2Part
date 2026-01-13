from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import trimesh
import tyro

sys.path.insert(0, './third_party/hunyuan3d/hy3dpaint')

from third_party.hunyuan3d.hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from third_party.hunyuan3d.torchvision_fix import fix_torchvision_functional_tensor

fix_torchvision_functional_tensor()

@dataclass
class Image2Texture:

    num_views: int = 6
    resolution: int = 512

    def init(self, device: torch.device) -> None:
        config = Hunyuan3DPaintConfig(max_num_view=self.num_views, resolution=self.resolution)
        config.device = device
        config.multiview_cfg_path = 'third_party/hunyuan3d/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml'
        config.realesrgan_ckpt_path = 'third_party/hunyuan3d/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'
        self.pipe = Hunyuan3DPaintPipeline(config)
        self.device = device

    def inference(
        self,
        mesh: Path,
        *,
        image: Path,
        remesh: bool = False,
    ) -> trimesh.Trimesh:
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh = self.pipe(
                mesh_path=str(mesh),
                image_path=str(image),
                output_mesh_path=str(Path(tmpdir) / 'temp.obj'),
                use_remesh=remesh,
            )
        return mesh

if __name__ == '__main__':

    def main(
        mesh: Path,
        image: Path,
        output: Path,
        remesh: bool = False,
    ) -> None:
        assert output.suffix in ['.obj', '.glb']
        model = Image2Texture()
        model.init('cuda')
        textured_mesh = model.inference(mesh=mesh, image=image, remesh=remesh)
        output.parent.mkdir(exist_ok=True, parents=True)
        textured_mesh.export(output)

    tyro.cli(main)
