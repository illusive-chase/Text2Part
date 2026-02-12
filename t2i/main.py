from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from diffusers import QwenImagePipeline
from PIL.Image import Image


@dataclass
class Text2Image:

    def init(self, device: torch.device) -> None:
        self.pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16).to(device)
        self.device = device

    def inference(
        self,
        prompt: str,
        *,
        aspect: Literal['1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3'] = '16:9',
        seed: int = 42,
        cfg: float = 4.0,
    ) -> Image:
        negative_prompt = (
            "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
        )
        width, height = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }[aspect]

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=cfg,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]
        return image

if __name__ == '__main__':

    def main(
        prompt: str,
        output: Path,
        seed: int = 42,
        aspect: Literal['1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3'] = '16:9',
        cfg: float = 4.0,
    ) -> None:
        model = Text2Image()
        model.init('cuda')
        image = model.inference(prompt, aspect=aspect, seed=seed, cfg=cfg)
        output.parent.mkdir(exist_ok=True, parents=True)
        image.save(output)

    tyro.cli(main)
