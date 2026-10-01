#!/bin/bash
# after the cleft sweep on GPU 0 finishes: real direction plus three random
# directions at layer 22, all-token steering, full precision, same items as the pilot
GPU=0
M=mistralai/Mistral-7B-Instruct-v0.3
echo "waiting for reports/steer_cleft_bf16.json since $(date)"
while [ ! -f reports/steer_cleft_bf16.json ]; do sleep 300; done
wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -ge 4000 ]; do sleep 300; done
}
wait_free
echo "real direction starting $(date)"
CUDA_VISIBLE_DEVICES=$GPU python steer_pilot.py --model $M --vectors vectors/mistral_bf16.pt \
  --layers 22 --positions all --no-4bit --out reports/steer_all_L22_rerun.json
for s in 1 2 3; do
  wait_free
  echo "random seed $s starting $(date)"
  CUDA_VISIBLE_DEVICES=$GPU python steer_pilot.py --model $M --vectors vectors/mistral_bf16.pt \
    --layers 22 --positions all --no-4bit --random-dir --seed $s \
    --out reports/random_all_L22_s$s.json
done
echo "all finished $(date)"
