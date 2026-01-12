import sys

import trimesh

sys.path.insert(0, './third_party/hunyuan3d/hy3dshape')
sys.path.insert(0, './third_party/hunyuan3d/hy3dpaint')

from third_party.hunyuan3d.hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from third_party.hunyuan3d.hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured: trimesh.Trimesh = shape_pipeline(image='temp.png')[0]
mesh_untextured.export('temp.glb')

if 0:
    paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
    mesh_textured = paint_pipeline('mesh_path', image_path='assets/demo.png')
