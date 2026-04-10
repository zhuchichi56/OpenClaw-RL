#!/bin/bash
# OpenClaw OEL — Phase 1: Experience Extraction
#
# Deploys the model as an API, collects interaction trajectories,
# and extracts experience items. No training happens in this phase.
#
# Architecture:
#   - 0 GPUs for Actor (no training)
#   - 1 GPU for Rollout (SGLang inference)
#   - 1 GPU for PRM (experience extraction)
#
# Usage:
#   EXP_NAME=oel-openclaw-q3-4b-extract-round1 \
#   SEED=42 \
#   bash run_qwen3_4b_openclaw_oel_extract.sh

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-1}

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# Model paths
HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3-4B}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

# Experiment config
EXP_NAME=${EXP_NAME:-"oel-openclaw-q3-4b-extract"}
SEED=${SEED:-42}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export TP="${TP:-1}"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"

# OEL Extract settings
export OPENCLAW_OEL_MODE="extract"
export OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH="${OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH:-8192}"
export OPENCLAW_RECORD_ENABLED="1"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/${EXP_NAME}_seed${SEED}_record.jsonl"

# Optional: Load experience from a previous round
# export OPENCLAW_OEL_EXPERIENCE_PATH="/tmp/oel-round1/experience_list.txt"

# Output directory for extracted experience
EXTRACT_OUTPUT="/tmp/${EXP_NAME}/global_step_${SEED}/extract_100_samples/experiences"
mkdir -p "${EXTRACT_OUTPUT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
   --save "/tmp/${EXP_NAME}/ckpt"
   --save-interval 999999
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_oel_rollout.generate_rollout_openclaw_oel

   --num-rollout 100000000
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --gradient-checkpointing
)

OEL_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path oel_distillation_loss.oel_distillation_loss_function
   --distill-topk 50
   --disable-compute-advantages-and-returns
   --disable-rewards-normalization
   --entropy-coef 0.01
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

LORA_ARGS=(
   --use-lora
   --lora-rank 16
   --lora-alpha 32
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP}"
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_TP:-${TP}}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_oel_api_server.generate
   --custom-rm-path openclaw_oel_api_server.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

echo "=========================================="
echo " OEL Phase 1: Experience Extraction"
echo " exp_name=${EXP_NAME}"
echo " seed=${SEED}"
echo " model=${HF_CKPT}"
echo " output=${EXTRACT_OUTPUT}"
echo "=========================================="

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${OEL_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]} \
   ${LORA_ARGS[@]}
