#!/bin/bash
# OpenClaw OEL — Online Mode (OPCD-compatible), Megatron Backend, 4 GPU
#
# Megatron full-parameter training (no LoRA — LoRA is FSDP-only).
# Uses bridge mode to load HF checkpoint directly into Megatron.
#
# Architecture:
#   - 2 GPUs for Actor (Megatron TP=2)
#   - 1 GPU for Rollout (SGLang)
#   - 1 GPU for PRM/Teacher (experience extraction + teacher logprobs)
#
# Requires isolated venv at /home/v-hezhu2/oel-megatron-env/.venv with:
#   torch, transformer-engine[pytorch,core_cu12], megatron-bridge==0.2.0rc6, mbridge
#
# Usage:
#   cd slime
#   HF_CKPT=/path/to/Qwen3-4B bash ../openclaw-oel/run_qwen3_4b_openclaw_oel_online_megatron.sh

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

# ---------- Isolated venv ----------
VENV_ROOT="/home/v-hezhu2/oel-megatron-env/.venv"
export PATH="${VENV_ROOT}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_ROOT}"
# Verify
echo "Python: $(which python3) -> $(python3 --version)"

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
MEGATRON_ROOT="${REPO_ROOT}/Megatron-LM"

# ---------- Model config for Megatron ----------
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-/workspace/OpenClaw-RL/saves/openclaw/qwen3-4b-oel}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_4b_oel_megatron_record.jsonl"
export TP="${TP:-1}"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"

# OEL settings
export OPENCLAW_OEL_MODE="online"
export OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH="${OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH:-2048}"

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   --rotary-base 1000000
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
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
   --log-probs-chunk-size 1024
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
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
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
   --custom-generate-function-path openclaw_oel_api_server.generate
   --custom-rm-path openclaw_oel_api_server.reward_func
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

USE_WANDB=${USE_WANDB:-0}
WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY:-${WANDB_API_KEY:-}}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project openclaw_rl
    --wandb-group qwen3-4b-openclaw-oel-megatron
    --wandb-key "${WANDB_KEY:-${WANDB_API_KEY}}"
  )
fi

export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${OEL_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}
