#!/usr/bin/env bash
set -euo pipefail

BASE="/root/code-assistant"
TRAINING_DIR="${BASE}/training"
EVAL_DIR="${BASE}/eval"

SFT_PID_FILE="${TRAINING_DIR}/sft_train_full_0319.pid"
SFT_LOG_FILE="${TRAINING_DIR}/sft_train_full_0319.log"
SFT_OUT_DIR="${TRAINING_DIR}/sft_ckpt_full_0319"

DPO_PID_FILE="${TRAINING_DIR}/dpo_train_full_0319.pid"
DPO_LOG_FILE="${TRAINING_DIR}/dpo_train_full_0319.log"
DPO_OUT_DIR="${TRAINING_DIR}/dpo_ckpt_full_0319"

EVAL_PID_FILE="${EVAL_DIR}/eval_full_0319.pid"
EVAL_LOG_FILE="${EVAL_DIR}/eval_full_0319.log"
EVAL_OUT_DIR="${EVAL_DIR}/results_full_0319"

PIPELINE_LOG="${TRAINING_DIR}/pipeline_full_0319.log"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "${PIPELINE_LOG}"; }

last_progress_line() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "(log_missing)"
    return 0
  fi
  # tqdm writes carriage returns; normalize to lines and keep the newest progress-ish line.
  tail -n 200 "$f" 2>/dev/null \
    | tr '\r' '\n' \
    | grep -E '([0-9]{1,3}%\\|)|(Total optimization steps)|(^\\[[A-Z]+\\|)' \
    | tail -n 1 \
    | head -c 400 || true
}

gpu_line() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | awk -F',' '{gsub(/ /,"",$0); print $1"MiB/"$2"MiB util="$3"%"}' || true
  fi
}

pid_is_running() {
  local pid_file="$1"
  if [ ! -f "$pid_file" ]; then
    return 1
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null | tr -d '\r\n' || true)"
  if [ -z "$pid" ]; then
    return 1
  fi
  ps -p "$pid" >/dev/null 2>&1
}

wait_for_stop() {
  local pid_file="$1"
  local log_file="$2"
  local name="$3"

  while pid_is_running "$pid_file"; do
    log "${name} running. gpu=$(gpu_line) progress=$(last_progress_line "$log_file")"
    sleep 120
  done
  log "${name} stopped. progress=$(last_progress_line "$log_file")"
}

ensure_dir() { mkdir -p "$1"; }

log "pipeline start"
ensure_dir "${TRAINING_DIR}"
ensure_dir "${EVAL_DIR}"

if pid_is_running "${SFT_PID_FILE}"; then
  log "SFT already running (pid_file=${SFT_PID_FILE})."
else
  log "SFT not running; starting via scripts/restart_sft.sh"
  bash "${BASE}/scripts/restart_sft.sh" >> "${PIPELINE_LOG}" 2>&1 || true
fi

wait_for_stop "${SFT_PID_FILE}" "${SFT_LOG_FILE}" "SFT"

if [ -d "${SFT_OUT_DIR}" ]; then
  log "SFT output dir exists: ${SFT_OUT_DIR}"
else
  log "SFT output dir missing: ${SFT_OUT_DIR}"
fi

if pid_is_running "${DPO_PID_FILE}"; then
  log "DPO already running (pid_file=${DPO_PID_FILE})."
else
  log "Starting DPO via scripts/restart_dpo.sh"
  bash "${BASE}/scripts/restart_dpo.sh" >> "${PIPELINE_LOG}" 2>&1 || true
fi

wait_for_stop "${DPO_PID_FILE}" "${DPO_LOG_FILE}" "DPO"

if [ -d "${DPO_OUT_DIR}" ]; then
  log "DPO output dir exists: ${DPO_OUT_DIR}"
else
  log "DPO output dir missing: ${DPO_OUT_DIR}"
fi

if pid_is_running "${EVAL_PID_FILE}"; then
  log "Eval already running (pid_file=${EVAL_PID_FILE})."
else
  log "Starting eval via scripts/restart_eval10.sh"
  bash "${BASE}/scripts/restart_eval10.sh" >> "${PIPELINE_LOG}" 2>&1 || true
fi

wait_for_stop "${EVAL_PID_FILE}" "${EVAL_LOG_FILE}" "EVAL"

if [ -f "${EVAL_OUT_DIR}/leaderboard.json" ]; then
  log "Eval done. leaderboard.json present: ${EVAL_OUT_DIR}/leaderboard.json"
else
  log "Eval done but leaderboard.json missing: ${EVAL_OUT_DIR}/leaderboard.json"
fi

log "pipeline finished"
