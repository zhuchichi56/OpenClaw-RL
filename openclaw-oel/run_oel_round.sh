#!/bin/bash
# OpenClaw OEL — Multi-Round Iterative Training
#
# Orchestrates the full OEL pipeline across multiple rounds:
#   Round N:
#     Phase 1: Extract — deploy model, collect trajectories, extract experience
#     Phase 2: Deploy  — collect deployment trajectories for consolidation
#     Phase 3: Consolidate — distillation training with experience-augmented teacher
#   Round N+1: uses Round N's checkpoint as starting point
#
# Usage:
#   # Round 1 (from base model)
#   ROUND=1 MODEL_PATH=models/Qwen3-4B bash run_oel_round.sh
#
#   # Round 2 (from Round 1 checkpoint)
#   ROUND=2 MODEL_PATH=/tmp/oel-openclaw-q3-4b-consolidate-round1/ckpt/latest \
#     bash run_oel_round.sh
#
# Configuration via environment variables:
#   ROUND           - Round number (default: 1)
#   MODEL_PATH      - Path to model (HF format)
#   EXP_PREFIX      - Experiment name prefix (default: oel-openclaw-q3-4b)
#   EXTRACT_SEEDS   - Comma-separated seeds for extraction (default: 50,100,150,200,250)
#   NUM_GPUS        - Total GPUs (default: 4)

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ==============================
# Configuration
# ==============================
ROUND=${ROUND:-1}
EXP_PREFIX=${EXP_PREFIX:-"oel-openclaw-q3-4b"}
MODEL_PATH=${MODEL_PATH:?"MODEL_PATH must be set"}
NUM_GPUS=${NUM_GPUS:-4}

# Extract seeds (each seed = one parallel extraction run)
EXTRACT_SEEDS=${EXTRACT_SEEDS:-"50,100,150,200,250"}
VAL_SAMPLES_LIMIT=${VAL_SAMPLES_LIMIT:-100}
VAL_SAMPLES_USE=${VAL_SAMPLES_USE:-50}

# Derived names
EXTRACT_EXP="${EXP_PREFIX}-extract-round${ROUND}"
DEPLOY_EXP="${EXP_PREFIX}-deploy-round${ROUND}"
CONSOLIDATE_EXP="${EXP_PREFIX}-consolidate-round${ROUND}"

echo "=================================================================="
echo " OEL Round ${ROUND}"
echo " model: ${MODEL_PATH}"
echo " extract_exp: ${EXTRACT_EXP}"
echo " deploy_exp: ${DEPLOY_EXP}"
echo " consolidate_exp: ${CONSOLIDATE_EXP}"
echo " seeds: ${EXTRACT_SEEDS}"
echo "=================================================================="

# ==============================
# Phase 1: Experience Extraction
# ==============================
echo ""
echo "===== Phase 1: Experience Extraction ====="

IFS=',' read -ra SEEDS <<< "${EXTRACT_SEEDS}"
for SEED in "${SEEDS[@]}"; do
    echo "--- Extracting with seed=${SEED} ---"
    EXP_NAME="${EXTRACT_EXP}" \
    SEED="${SEED}" \
    HF_CKPT="${MODEL_PATH}" \
    NUM_GPUS="${NUM_GPUS}" \
    bash "${SCRIPT_DIR}/run_qwen3_4b_openclaw_oel_extract.sh"
done

# Build experience list
echo "--- Building experience list ---"
python "${SCRIPT_DIR}/tools/make_exp_list.py" \
    --exp-name "${EXTRACT_EXP}" \
    --ckpt-start "${SEEDS[0]}" \
    --ckpt-end "${SEEDS[-1]}" \
    --ckpt-step $(( SEEDS[1] - SEEDS[0] )) \
    --val-samples-limit "${VAL_SAMPLES_LIMIT}" \
    --val-samples-use "${VAL_SAMPLES_USE}" \
    --base-dir /tmp

EXP_LIST="/tmp/${EXTRACT_EXP}/experience_list.txt"
echo "Experience list: ${EXP_LIST}"
cat "${EXP_LIST}"

# ==============================
# Phase 2: Deploy / Trajectory Collection
# ==============================
echo ""
echo "===== Phase 2: Deploy / Trajectory Collection ====="

EXP_NAME="${DEPLOY_EXP}" \
HF_CKPT="${MODEL_PATH}" \
NUM_GPUS="${NUM_GPUS}" \
bash "${SCRIPT_DIR}/run_qwen3_4b_openclaw_oel_deploy.sh"

DEPLOY_DIR="/tmp/${DEPLOY_EXP}/deploy_data"
echo "Deploy data: ${DEPLOY_DIR}"
ls -la "${DEPLOY_DIR}" 2>/dev/null || echo "(no deploy data files yet)"

# ==============================
# Phase 3: Consolidation Training
# ==============================
echo ""
echo "===== Phase 3: Consolidation Training ====="

EXP_NAME="${CONSOLIDATE_EXP}" \
EXP_PATH="${EXP_LIST}" \
DEPLOY_SAVE_DIR="${DEPLOY_DIR}" \
HF_CKPT="${MODEL_PATH}" \
NUM_GPUS="${NUM_GPUS}" \
bash "${SCRIPT_DIR}/run_qwen3_4b_openclaw_oel_consolidate.sh"

# ==============================
# Summary
# ==============================
CKPT_DIR="/tmp/${CONSOLIDATE_EXP}/ckpt"
echo ""
echo "=================================================================="
echo " OEL Round ${ROUND} Complete!"
echo " Checkpoint: ${CKPT_DIR}"
echo ""
echo " To start Round $((ROUND+1)):"
echo "   ROUND=$((ROUND+1)) MODEL_PATH=${CKPT_DIR}/latest bash run_oel_round.sh"
echo "=================================================================="
