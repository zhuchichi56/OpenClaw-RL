#!/bin/bash
# OpenClaw Combined-OPCD — Qwen3-1.7B Full-Parameter (No LoRA), Megatron backend
#
# Combined-OPCD = GRPO (RL) + OPCD (experience-augmented distillation)
# Unlike Combined (GRPO+OPD), this uses OEL's experience accumulation
# instead of OPD's per-turn hint mechanism.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-1}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
OEL_DIR="$(cd -- "${SCRIPT_DIR}/../openclaw-oel" &>/dev/null && pwd)"

# Source model architecture args for Megatron backend
source "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3-1.7B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/saves/openclaw/qwen3-1.7b-combine-opcd}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-1.7b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_1.7b_combine_opcd_record.jsonl"
export TP="${TP:-1}"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"
# Teacher (PRM) weight update: "0" (frozen, default) or "1" (co-evolving)
export OPENCLAW_UPDATE_PRM_WEIGHTS="${OPENCLAW_UPDATE_PRM_WEIGHTS:-0}"
# Combined weights
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
# OEL experience settings
export OPENCLAW_OEL_MODE="${OPENCLAW_OEL_MODE:-online}"
export OPENCLAW_OEL_EXTRACTION_PROMPT="${OPENCLAW_OEL_EXTRACTION_PROMPT:-v5}"
export OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH="${OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH:-2048}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 5
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_opcd_rollout.generate_rollout_openclaw_combine_opcd

   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
)

COMBINE_OPCD_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --loss-type custom_loss
   --custom-loss-function-path combine_opcd_loss.combine_loss_function
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

EVAL_ARGS=()

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
   --custom-generate-function-path openclaw_combine_opcd_api_server.generate
   --custom-rm-path openclaw_combine_opcd_api_server.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM/:${SCRIPT_DIR}:${OEL_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE:-1}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\",
    \"OPENCLAW_UPDATE_PRM_WEIGHTS\": \"${OPENCLAW_UPDATE_PRM_WEIGHTS}\",
    \"OPENCLAW_OEL_MODE\": \"${OPENCLAW_OEL_MODE}\",
    \"OPENCLAW_OEL_EXTRACTION_PROMPT\": \"${OPENCLAW_OEL_EXTRACTION_PROMPT}\",
    \"OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH\": \"${OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH}\"
  }
}"

cd "${SCRIPT_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --working-dir="${SCRIPT_DIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${COMBINE_OPCD_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}
