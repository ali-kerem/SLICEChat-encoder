#!/bin/bash
set -euo pipefail

run_name="slicechat"
base_log_dir="logs/mae"

# You can set the devices and number of gpus
NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
repo_root="$(pwd)"
source .venv/bin/activate

echo "Starting ${run_name}"

get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

master_addr="127.0.0.1"
nnodes=1
job_id="${RANDOM}"
master_port=$(get_free_port)

echo "Run name: ${run_name}"
echo "Master address: ${master_addr}"
echo "Master port: ${master_port}"
echo "Nnodes: ${nnodes}"
echo "Job ID: ${job_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

node_rank=0

log_dir="${base_log_dir}/${run_name}"
mkdir -p "${log_dir}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${master_addr}:${master_port}" \
    --rdzv_id="${job_id}" \
    -m slicechat_encoder.train.mae.main_pretrain \
    --distributed \
    --model_config src/slicechat_encoder/models/mamba/model_configs/base.yaml \
    --batch_size 4 \
    --epochs 30 \
    --mask_ratio 0.4 \
    --data_path /path/to/wsi-tensors/ \
    --csv_path data/mae_slidechat.csv \
    --seed 0 \
    --deterministic \
    --num_workers 4 \
    --output_dir "${log_dir}" \
    --log_dir "${log_dir}" \
    2>&1 | tee "${log_dir}/full.log"
    # --wandb \
    # --wandb_project "slicechat-encoder" \
    # --wandb_run_name "${run_name}" \
