#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lf

ROOT=/root/code-assistant
EVAL_OUT=${ROOT}/eval/results_reg45_v4_0319
EVAL_LOG=${ROOT}/eval/eval_reg45_v4_0319.log
EVAL_PID=${ROOT}/eval/eval_reg45_v4_0319.pid
TARGET_ADAPTER=${ROOT}/training/dpo_ckpt_reg20_v4b_0319/adapter_model.safetensors

while [ ! -f "${TARGET_ADAPTER}" ]; do
  sleep 30
done

mkdir -p "${EVAL_OUT}"
nohup python "${ROOT}/scripts/eval_step10.py" \
  --base-model "${ROOT}/models/Qwen2.5-Coder-3B-Instruct" \
  --sft-adapter "${ROOT}/training/sft_ckpt_reg20_v3_0319" \
  --dpo-adapter "${ROOT}/training/dpo_ckpt_reg20_v4b_0319" \
  --dpo-pairs "${ROOT}/data/processed_v4/dpo_train_func_completion.jsonl" \
  --out-dir "${EVAL_OUT}" \
  --num-humaneval 45 \
  --num-pref 100 \
  --pass-k 3 \
  > "${EVAL_LOG}" 2>&1 &
echo $! > "${EVAL_PID}"
echo "eval_started $(cat "${EVAL_PID}")"
