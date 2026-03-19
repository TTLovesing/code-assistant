#!/usr/bin/env bash
set -e
pkill -9 -f train_dpo_pyfunc.yaml || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lf
cd /root/code-assistant/LLaMA-Factory
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train /root/code-assistant/train_dpo_pyfunc.yaml > /root/code-assistant/training/dpo_train_full_0319.log 2>&1 &
echo $! > /root/code-assistant/training/dpo_train_full_0319.pid
echo STARTED
