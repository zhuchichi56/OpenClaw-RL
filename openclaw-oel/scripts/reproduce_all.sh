#!/bin/bash
# =============================================================================
# Reproduce all 4 experiments from the paper.
#
# Experiments (paper naming):
#   1. SDFT/SDPO-style — per-turn distillation without experience (baseline)
#   2. OPCD            — On-Policy Context Distillation, cross-session experience accumulation
#   3. OPCD-Pre        — OPCD + before-the-fact per-turn experience extraction
#   4. OPCD-Suc        — OPCD + after-the-fact successive (post-hoc) experience extraction + replay
#
# Prerequisites:
#   - Docker container "openclaw-slime" running with 4x H100 GPUs
#   - Model weights at ${REPO_ROOT}/models/Qwen3-1.7B
#   - GPT-4.1 API key set via OPENAI_API_KEY environment variable
#
# Usage:
#   bash scripts/reproduce_all.sh              # run all 4 experiments
#   bash scripts/reproduce_all.sh --only sdpo    # run only SDFT/SDPO-style
#   bash scripts/reproduce_all.sh --only opcd-suc # run only OPCD-Suc
#
# Each experiment: ~2 hours (40 rounds, eval-every 2)
# Total: ~8 hours sequential
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OEL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${OEL_DIR}/.." &>/dev/null && pwd)"
OPD_DIR="${REPO_ROOT}/openclaw-opd"

# Default: run all experiments
ONLY="${1:-all}"
if [[ "$1" == "--only" ]]; then
    ONLY="${2:-all}"
fi

TRAINING_ROUNDS=${TRAINING_ROUNDS:-40}
EVAL_EVERY=${EVAL_EVERY:-2}
SCENARIO=${SCENARIO:-student}

# Common eval args
EVAL_ARGS=(
    --method oel
    --training-rounds "${TRAINING_ROUNDS}"
    --eval-every "${EVAL_EVERY}"
    --scenario "${SCENARIO}"
    --problem-file ../data/hard_problems_train.json
    --eval-problem-file ../data/hard_problems_eval.json
)

# ---------------------------------------------------------------------------
# Helper: wait for training server to be ready
# ---------------------------------------------------------------------------
wait_for_server() {
    local url="${1:-http://localhost:30000/healthz}"
    local max_wait="${2:-300}"  # 5 minutes
    echo "  Waiting for training server at ${url} ..."
    for i in $(seq 1 "${max_wait}"); do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: server not ready after ${max_wait}s"
    return 1
}

# ---------------------------------------------------------------------------
# Helper: stop training backend
# ---------------------------------------------------------------------------
stop_training() {
    echo "  Stopping training backend..."
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    echo "  Training backend stopped."
}

# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------
run_experiment() {
    local name="$1"
    local train_dir="$2"
    local train_script="$3"
    local eval_dir="$4"
    local method="$5"
    shift 5
    local env_vars=("$@")

    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  Train: ${train_dir}/${train_script}"
    echo "  Eval:  ${eval_dir}/"
    echo "  Env:   ${env_vars[*]:-<none>}"
    echo "================================================================"

    # Stop any previous training
    stop_training

    # Start training backend
    echo "  Starting training backend..."
    cd "${train_dir}"
    env "${env_vars[@]}" bash "${train_script}" > "/tmp/reproduce_${name}_train.log" 2>&1 &
    TRAIN_PID=$!

    # Wait for server
    if ! wait_for_server "http://localhost:30000/healthz" 300; then
        # Try /v1/chat/completions as fallback (some servers don't have /healthz)
        wait_for_server "http://localhost:30000/v1/chat/completions" 60 || {
            echo "  FATAL: training server failed to start. See /tmp/reproduce_${name}_train.log"
            return 1
        }
    fi

    # Run evaluation
    echo "  Starting evaluation..."
    cd "${eval_dir}"
    mkdir -p results
    python3 -u gsm8k_personal_agent.py \
        --method "${method}" \
        --training-rounds "${TRAINING_ROUNDS}" \
        --eval-every "${EVAL_EVERY}" \
        --scenario "${SCENARIO}" \
        --problem-file ../data/hard_problems_train.json \
        --eval-problem-file ../data/hard_problems_eval.json \
        2>&1 | tee "/tmp/reproduce_${name}_eval.log"

    # Rename result file to include experiment variant for auto-detection
    local latest
    latest=$(ls -t "${eval_dir}/results/personal_agent_${method}_"*.json 2>/dev/null | head -1)
    if [[ -n "${latest}" ]]; then
        local base dir_name
        dir_name=$(dirname "${latest}")
        base=$(basename "${latest}" .json)
        cp "${latest}" "${dir_name}/${base}_${name}.json"
        echo "  Result tagged: ${dir_name}/${base}_${name}.json"
    fi

    echo "  Experiment ${name} DONE."
    echo "  Results: ${eval_dir}/results/"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "============================================"
echo "  OpenClaw-RL: 4-Method Reproduction"
echo "  (SDFT/SDPO-style, OPCD, OPCD-Pre, OPCD-Suc)"
echo "  Rounds: ${TRAINING_ROUNDS}, Eval-every: ${EVAL_EVERY}"
echo "  Target: ${ONLY}"
echo "============================================"

if [[ "${ONLY}" == "all" || "${ONLY}" == "sdpo" ]]; then
    run_experiment "sdpo" \
        "${OPD_DIR}" "run_qwen3_1.7b_openclaw_opd_topk.sh" \
        "${OPD_DIR}/eval" "opd"
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "opcd" ]]; then
    run_experiment "opcd" \
        "${OEL_DIR}" "run_qwen3_1.7b_openclaw_oel_online.sh" \
        "${OEL_DIR}/eval" "oel" \
        "OPENCLAW_OEL_SESSION_EXPERIENCE=0"
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "opcd-pre" ]]; then
    run_experiment "opcd_pre" \
        "${OEL_DIR}" "run_qwen3_1.7b_openclaw_oel_online.sh" \
        "${OEL_DIR}/eval" "oel" \
        "OPENCLAW_OEL_SESSION_EXPERIENCE=1"
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "opcd-suc" ]]; then
    run_experiment "opcd_suc" \
        "${OEL_DIR}" "run_qwen3_1.7b_openclaw_oel_online.sh" \
        "${OEL_DIR}/eval" "oel" \
        "OPENCLAW_OEL_SESSION_EXPERIENCE=replay"
fi

# Stop training at the end
stop_training

echo ""
echo "============================================"
echo "  ALL DONE. Results in:"
echo "    SDFT/SDPO-style: ${OPD_DIR}/eval/results/"
echo "    OPCD:            ${OEL_DIR}/eval/results/"
echo "    OPCD-Pre:        ${OEL_DIR}/eval/results/"
echo "    OPCD-Suc:        ${OEL_DIR}/eval/results/"
echo ""
echo "  Generate comparison plot:"
echo "    python3 ${OEL_DIR}/scripts/plot_3method_comparison.py"
echo "============================================"
