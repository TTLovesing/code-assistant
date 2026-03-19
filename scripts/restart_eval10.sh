#!/usr/bin/env bash
set -e
pkill -9 -f eval_step10.py || true
source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/code-assistant/.hf
export HF_DATASETS_CACHE=/root/code-assistant/data/raw/hf_datasets_cache
mkdir -p /root/code-assistant/eval/results_full_0319_postproc_rerun
cd /root/code-assistant
nohup python /root/code-assistant/scripts/eval_step10.py \
  --base-model /root/code-assistant/models/Qwen2.5-Coder-3B-Instruct \
  --sft-adapter /root/code-assistant/training/sft_ckpt_full_0319 \
  --dpo-adapter /root/code-assistant/training/dpo_ckpt_full_0319 \
  --dpo-pairs /root/code-assistant/data/processed_v2/dpo_train_func_completion.jsonl \
  --out-dir /root/code-assistant/eval/results_full_0319_postproc_rerun \
  --num-humaneval 20 \
  --num-pref 100 \
  --pass-k 3 \
  > /root/code-assistant/eval/eval_full_0319_postproc_rerun.log 2>&1 &
echo $! > /root/code-assistant/eval/eval_full_0319_postproc_rerun.pid
echo STARTED
