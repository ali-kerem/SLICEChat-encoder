#!/bin/bash
set -euo pipefail

run_name="slicechat_clip"
base_log_dir="logs/clip"

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
    -m slicechat_encoder.train.open_clip.open_clip_train.main \
    --resume "logs/mae/slicechat/checkpoints/checkpoint-29.pth" \
    --pruner-train cropr visual.pooler text.proj \
    --pruner-train-cropr-cfg "src/slicechat_encoder/models/token_compressors/cropr_configs/base.yaml" \
    --pruner-train-pooler-type "attention" \
    --pruner-train-pruning-rate 0.8 \
    --train-data="data/generated/clip_slidechat_train.json" \
    --val-data="data/generated/clip_slidechat_test.json" \
    --train-data-dir="/path/to/wsi-tensors" \
    --val-data-dir="/path/to/wsi-tensors" \
    --model base \
    --seed 0 \
    --deterministic \
    --epochs=20 \
    --lr=7e-4 \
    --wd=0.1 \
    --warmup 0.1 \
    --batch-size=32 \
    --accum-freq 1 \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --workers=4 \
    --zeroshot-frequency 1 \
    --precision=bf16 \
    --log-every-n-steps=1 \
    --delete-previous-checkpoint \
    --name "${run_name}" \
    --log-dir "${base_log_dir}" \
    2>&1 | tee "${log_dir}/full.log"
    # --report-to wandb \
    # --wandb-run-name "${run_name}" \
    # --wandb-project-name "slicechat-encoder" \
