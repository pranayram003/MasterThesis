#!/bin/bash
# usage: bash queue_cluster.sh GPU
# waits until the given GPU has less than 4000 MiB in use, then runs the
# clustering check (both stages). Log goes to reports/cluster_check.log
GPU=${1:-0}
echo "waiting for GPU $GPU since $(date)"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  if [ "$used" -lt 4000 ]; then break; fi
  sleep 300
done
echo "starting on GPU $GPU at $(date)"
CUDA_VISIBLE_DEVICES=$GPU python cluster_check.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --vectors vectors/mistral_bf16.pt \
  --items data/para_orig.jsonl data/para_never.jsonl data/para_cleft.jsonl \
  --n 200
echo "finished at $(date)"
