#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lf

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/code-assistant/.hf
export HF_DATASETS_CACHE=/root/code-assistant/data/raw/hf_datasets_cache

ROOT=/root/code-assistant
LOG_DIR=${ROOT}/training
EVAL_DIR=${ROOT}/eval

DPO_LOG=${LOG_DIR}/dpo_train_reg20_v4_0319.log
DPO_PID=${LOG_DIR}/dpo_train_reg20_v4_0319.pid
EVAL_LOG=${EVAL_DIR}/eval_reg50_v4_0319.log
EVAL_PID=${EVAL_DIR}/eval_reg50_v4_0319.pid
EVAL_OUT=${EVAL_DIR}/results_reg50_v4_0319

cd ${ROOT}

nohup llamafactory-cli train ${ROOT}/train_dpo_pyfunc_reg20_v4_0319.yaml > ${DPO_LOG} 2>&1 &
echo $! > ${DPO_PID}
echo "DPO_STARTED $(cat ${DPO_PID})"

while true; do
  if ! kill -0 "$(cat ${DPO_PID})" 2>/dev/null; then
    break
  fi
  sleep 30
done
echo "DPO_DONE"

mkdir -p ${EVAL_OUT}
nohup python ${ROOT}/scripts/eval_step10.py \
  --base-model ${ROOT}/models/Qwen2.5-Coder-3B-Instruct \
  --sft-adapter ${ROOT}/training/sft_ckpt_reg20_v3_0319 \
  --dpo-adapter ${ROOT}/training/dpo_ckpt_reg20_v4_0319 \
  --dpo-pairs ${ROOT}/data/processed_v4/dpo_train_func_completion.jsonl \
  --out-dir ${EVAL_OUT} \
  --num-humaneval 50 \
  --num-pref 100 \
  --pass-k 3 \
  > ${EVAL_LOG} 2>&1 &
echo $! > ${EVAL_PID}
echo "EVAL_STARTED $(cat ${EVAL_PID})"
