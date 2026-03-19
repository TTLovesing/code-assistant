#!/usr/bin/env bash
set -euo pipefail

BASE="/root/code-assistant"
TRAIN_DIR="${BASE}/training"
EVAL_DIR="${BASE}/eval"
LOG="${TRAIN_DIR}/pipeline_reg20_v3_0319.log"

SFT_PID="${TRAIN_DIR}/sft_train_reg20_v3_0319.pid"
SFT_LOG="${TRAIN_DIR}/sft_train_reg20_v3_0319.log"

DPO_PID="${TRAIN_DIR}/dpo_train_reg20_v3_0319.pid"
DPO_LOG="${TRAIN_DIR}/dpo_train_reg20_v3_0319.log"

EVAL_PID="${EVAL_DIR}/eval_reg20_v3_0319.pid"
EVAL_LOG="${EVAL_DIR}/eval_reg20_v3_0319.log"
EVAL_OUT="${EVAL_DIR}/results_reg20_v3_0319"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

pid_alive() {
  local f="$1"
  [ -f "$f" ] || return 1
  local p
  p="$(cat "$f" 2>/dev/null || true)"
  [ -n "$p" ] || return 1
  ps -p "$p" >/dev/null 2>&1
}

progress_line() {
  local f="$1"
  [ -f "$f" ] || { echo "(log_missing)"; return 0; }
  tail -n 200 "$f" | tr '\r' '\n' | grep -E '([0-9]{1,3}%\|)|(^\[[A-Z]+\|)|(^\[[a-z]+\])' | tail -n 1 | head -c 280 || true
}

wait_stage() {
  local name="$1"
  local pidf="$2"
  local logf="$3"
  while pid_alive "$pidf"; do
    say "${name} running: $(progress_line "$logf")"
    sleep 90
  done
  say "${name} done: $(progress_line "$logf")"
}

say "pipeline start"

wait_stage "SFT" "$SFT_PID" "$SFT_LOG"

say "start DPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lf
cd /root/code-assistant/LLaMA-Factory
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train /root/code-assistant/train_dpo_pyfunc_reg20_v3_0319.yaml > "${DPO_LOG}" 2>&1 &
echo $! > "${DPO_PID}"
sleep 2
wait_stage "DPO" "$DPO_PID" "$DPO_LOG"

say "start EVAL"
mkdir -p "${EVAL_OUT}"
cd /root/code-assistant
nohup python /root/code-assistant/scripts/eval_step10.py \
  --base-model /root/code-assistant/models/Qwen2.5-Coder-3B-Instruct \
  --sft-adapter /root/code-assistant/training/sft_ckpt_reg20_v3_0319 \
  --dpo-adapter /root/code-assistant/training/dpo_ckpt_reg20_v3_0319 \
  --dpo-pairs /root/code-assistant/data/processed_v3/dpo_train_func_completion.jsonl \
  --out-dir "${EVAL_OUT}" \
  --num-humaneval 20 \
  --num-pref 100 \
  --pass-k 3 > "${EVAL_LOG}" 2>&1 &
echo $! > "${EVAL_PID}"
sleep 2
wait_stage "EVAL" "$EVAL_PID" "$EVAL_LOG"

if [ -f "${EVAL_OUT}/leaderboard.json" ]; then
  say "pipeline done, leaderboard ready: ${EVAL_OUT}/leaderboard.json"
else
  say "pipeline done, leaderboard missing"
fi

