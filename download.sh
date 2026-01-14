export HF_ENDPOINT=https://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
hf download tencent/Hunyuan3D-2.1 --repo-type model
hf download Qwen/Qwen-Image-2512 --repo-type model
hf download facebook/dinov2-giant --repo-type model
mkdir ~/.u2net && wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O ~/.u2net/u2net.onnx
hf download tencent/Hunyuan3D-Part --repo-type model
hf download facebook/sonata --repo-type model
ln -s .../.cache/Hunyuan3D-Part/xpart.pt ~/.cache/xpart/tencent/Hunyuan3D-Part/xpart.pt