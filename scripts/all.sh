unset http_proxy HTTP_PROXY HTTPS_PROXY https_proxy
export HF_ENDPOINT=https://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1

source .venv/bin/activate

# bash scripts/faucet.sh
bash scripts/fridge.sh
bash scripts/micro.sh
bash scripts/scissors.sh
bash scripts/stationery.sh
bash scripts/toolbox.sh
bash scripts/trash.sh
bash scripts/washing.sh