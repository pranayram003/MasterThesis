#!/bin/bash
# usage: wait_and_run.sh GPU VECTORS OUT
# polls the given GPU every 5 minutes and starts the cleft sweep once it is free
GPU=$1; VEC=$2; OUT=$3
echo "waiting for GPU $GPU since $(date)"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  if [ "$used" -lt 4000 ]; then break; fi
  sleep 300
done
echo "starting on GPU $GPU at $(date)"
CUDA_VISIBLE_DEVICES=$GPU python steer_para.py --model mistralai/Mistral-7B-Instruct-v0.3 \
  --vectors "$VEC" \
  --items data/para_orig.jsonl data/para_never.jsonl data/para_cleft.jsonl \
  --layers 22 --n 200 --out "$OUT"
echo "finished at $(date)"
