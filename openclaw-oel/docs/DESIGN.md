# OEL Design Document: Architecture, Differences from OPD, and Implementation Details

This document describes the design and implementation of the **openclaw-oel** module, which implements **Online Experiential Learning (OEL)** and **On-Policy Context Distillation (OPCD)** in a unified framework. It explains the key architectural differences from the existing **openclaw-opd** (On-Policy Distillation with Hindsight Hints) module, details the training pipeline, and documents the implementation choices.

**References**:
- [Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) (OEL)
- [On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) (OPCD)

---

## 1. Motivation: Why OEL over OPD

OPD (On-Policy Distillation) works per-turn: after each assistant response, a judge scores the response against the next state, extracts a hindsight hint, and uses that hint to construct a teacher signal. If the majority vote rejects the hint, the sample is dropped entirely. This approach has several limitations:

1. **High drop rate.** Judge voting is noisy. When `m=3` votes produce no majority positive, the entire turn's training signal is discarded. In practice, 30--60% of main-line turns yield no usable sample.

2. **Per-turn hints are narrow.** Each hint is conditioned on a single `(response, next_state)` pair. The model learns local fixes, not transferable patterns.

3. **Reward signal required.** The judge is effectively a binary reward model (good/bad). Without a discriminative signal, OPD cannot extract supervision.

OEL addresses all three issues:

1. **No samples dropped.** Every turn produces a training sample. The teacher is the same model with accumulated experience injected into the prompt. As long as experience exists, the teacher always provides a valid KL target.

2. **Transferable knowledge.** Experience accumulates across sessions. The extracted items are high-level, domain-general insights (e.g., "prioritize step-by-step reasoning for multi-step math"), not turn-specific patches.

3. **No reward signal.** Experience extraction uses a generative prompt (`EXPERIENCE_UPDATE_PROMPT_V4`), not a discriminative judge. The model self-reflects on its own interaction history to produce new experience items.

---

## 2. Conceptual Comparison: OPD vs. OEL/OPCD

| Aspect | OPD | OEL / OPCD |
|--------|-----|------------|
| **Teacher construction** | Same model + per-turn hindsight hint | Same model + accumulated session experience |
| **Hint/experience source** | Judge extracts hint from `(response, next_state)` | Model self-extracts experience from full session via `EXPERIENCE_UPDATE_PROMPT_V4` |
| **Sample acceptance** | Majority vote gate: if no positive hint, drop sample | All turns produce samples (teacher always has experience) |
| **Knowledge scope** | Single-turn, narrow, backward-looking | Cross-session, general, forward-looking |
| **Reward signal** | Required (judge votes +1/-1) | Not required |
| **Loss function** | PPO-style clipped policy loss + KL penalty (Option A) or reverse KL top-K (Option B) | Reverse KL top-K with tail trick (always) |
| **Training paradigm** | Online only (continuous) | Online (OPCD mode) or iterative (OEL 3-phase mode) |

### Key insight
OPD's teacher signal is a **hint-augmented prompt**: `[original_prompt + hint]`. OEL's teacher signal is an **experience-augmented prompt**: `[experience + original_prompt]`. Both distill knowledge from a stronger version of the same model (self-distillation), but the source of the augmentation differs fundamentally.

---

## 3. Architecture Overview

### 3.1 Unified Mode Switching

`openclaw-oel` supports four modes controlled by `OPENCLAW_OEL_MODE`:

```
OPENCLAW_OEL_MODE=online       # OPCD: continuous online training
OPENCLAW_OEL_MODE=extract      # OEL Phase 1: experience extraction (inference-only)
OPENCLAW_OEL_MODE=deploy       # OEL Phase 2: trajectory collection (inference-only)
OPENCLAW_OEL_MODE=consolidate  # OEL Phase 3: consolidation training
```

#### Online mode (= OPCD)
A single long-running process that serves requests via an OpenAI-compatible API proxy. On each completed session, experience is extracted and accumulated in-memory. For each main-line turn, the teacher prompt is `[experience + original_prompt]`, and the student prompt is `[original_prompt]`. Training samples are submitted continuously to the Slime trainer.

#### OEL iterative mode (Extract -> Deploy -> Consolidate)
Training is split into three explicit phases per round:

1. **Extract** -- Deploy the model, interact with environments, extract experience from completed sessions. No weight updates. Experience is saved as `experience_session_N.txt` files.

2. **Deploy** -- Deploy the model again, interact with environments, save full trajectories (prompt IDs + response IDs + logprobs) as JSON files. No weight updates.

3. **Consolidate** -- Load the saved trajectories and experience files. For each trajectory turn, construct teacher (model + experience) and student (model alone) prompts, compute teacher logprobs, and train with reverse-KL distillation.

Rounds iterate: the consolidated checkpoint becomes the starting model for the next round's extract phase.

### 3.2 Component Map

```
External Environment (OpenClaw)
        |
        | HTTP POST /v1/chat/completions
        v
+---------------------------+
| openclaw_oel_api_server   |    <-- FastAPI proxy
|  - Request handling       |
|  - Experience injection   |    teacher prompt = [exp; prompt]
|  - Teacher logprob query  |    via SGLang /generate (logprob_start_len)
|  - Experience extraction  |    via PRM engine (generative)
|  - Trajectory saving      |    (deploy mode: save to disk)
|  - Sample submission      |    -> output_queue -> rollout
+---------------------------+
        |
        v
+---------------------------+
| openclaw_oel_rollout      |    <-- Rollout bridge
|  - AsyncRolloutWorker     |
|  - Drain output queue     |
|  - Mode-aware dispatch    |
+---------------------------+
        |
        v
+---------------------------+
| Slime Trainer (FSDP)      |    <-- Training backend
|  - oel_distillation_loss  |    reverse KL top-K + tail trick
|  - LoRA updates           |
+---------------------------+
```

---

## 4. Implementation Details

### 4.1 Experience Accumulation

**Data structure**: A single string `_experience_text` containing `- EXPERIENCE ITEM: ...` lines. Maintained in memory, protected by `_experience_lock`.

**Extraction trigger**: When a session completes (`session_done=True` in the API request), the full conversation is sent to the PRM engine with `EXPERIENCE_UPDATE_PROMPT_V4`:

```
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction:
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task: Generate additional experience items ...
Rules:
- Format: "- EXPERIENCE ITEM: ..."
- General, high-level, widely applicable insights only
- Focus on: writing style, explanation quality, tool usage, user interaction
- If conflict with previous items, describe conflict and resolution
```

**Update formula**: Following OEL (arXiv:2603.16856), experience at session $i$ is:

$$e_i' \sim \pi_{\text{extract}}(\cdot \mid \tau_i, e_{i-1})$$

where $\tau_i$ is the session trajectory and $e_{i-1}$ is the previous experience. The new items $e_i'$ are appended to get $e_i = [e_{i-1}; e_i']$.

**Truncation**: FIFO truncation to `OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH` tokens (default: 2048). When truncating, the earliest items are dropped while preserving line boundaries (`- EXPERIENCE ITEM:` alignment). Implemented in `_truncate_experience()`.

**Persistence**: After each extraction, a cumulative snapshot is saved as `results/experiences/experience_session_{N}.txt`. In OEL extract mode, these files are the primary output.

### 4.2 Teacher Construction

The teacher is the **same model** (same weights, same SGLang endpoint) but with an experience-augmented prompt. The augmentation is done by injecting experience into the last user message:

```python
def _inject_experience_to_messages(self, messages, experience):
    """Inject into last user message."""
    # Find last user message
    # Replace content with:
    #   "Given previous learned experience:\n# Experience\n{experience}\n\n
    #    Apply the relevant experience to respond to the following:\n{original_content}"
```

This means:
- **Student input**: `[system, user_1, assistant_1, ..., user_k]` -- bare prompt
- **Teacher input**: `[system, user_1, assistant_1, ..., user_k_with_experience]` -- experience-augmented

The student and teacher share the same response tokens $y$. The teacher's logprobs on $y$ are computed via SGLang's `logprob_start_len` parameter (prefill-only, no generation):

```python
payload = {
    "input_ids": enhanced_ids,        # [teacher_prompt; response_tokens]
    "sampling_params": {"max_new_tokens": 0},
    "return_logprob": True,
    "logprob_start_len": len(enhanced_ids) - response_len,
}
```

For top-K distillation, `top_logprobs_num=K` is additionally set to retrieve the teacher's top-K token distributions per position.

### 4.3 Distillation Loss

**Reverse KL (student || teacher)** with top-K approximation:

$$\mathcal{L} = \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\text{KL}}\!\left(\pi_\theta(\cdot \mid x, y_{<t}) \;\|\; \pi_{\text{teacher}}(\cdot \mid e, x, y_{<t})\right)$$

Only the teacher's top-$K$ vocabulary tokens are considered. The remaining probability mass is captured by a tail bin:

$$D_{\text{KL}}^{K+1} = \sum_{k=1}^{K+1} p_s^{(k)} \left(\log p_s^{(k)} - \log p_t^{(k)}\right)$$

where the $(K+1)$-th bin uses:

$$\log p_{\text{tail}} = \log\!\left(1 - \exp\!\left(\text{logsumexp}(\log p_1, \ldots, \log p_K)\right)\right)$$

implemented via `log(-expm1(logsumexp(...)))` for numerical stability.

**Default**: $K=50$, with optional entropy bonus ($\beta_{\text{ent}}=0.01$).

**Why reverse KL, not forward KL?** Forward KL (`KL(teacher || student)`) is mode-covering and requires sampling from the teacher. Reverse KL (`KL(student || teacher)`) is mode-seeking and uses on-policy student samples. Key advantage: the student trains on its own distribution, preserving out-of-distribution (OOD) performance on tasks unrelated to the experience. This is critical for agentic settings where the model must generalize beyond its training distribution.

### 4.4 Comparison of Loss Functions: OPD vs OEL

| | OPD Option A (token-level) | OPD Option B (top-K) | OEL |
|--|---|---|---|
| **Loss** | PPO clipped + KL penalty | Reverse KL top-K | Reverse KL top-K |
| **Teacher signal** | Single token logprob advantage $A_t$ | Top-K distribution | Top-K distribution |
| **Advantage** | $\log \pi_T(a_t \mid s+\text{hint}) - \log \pi_\theta(a_t \mid s)$ | N/A (full distribution) | N/A (full distribution) |
| **Sample gating** | Majority vote (drop if no positive hint) | Majority vote (drop if no positive hint) | No gating (all samples used) |
| **External loss** | No (built-in PPO) | Yes (`topk_distillation_loss.py`) | Yes (`oel_distillation_loss.py`) |

OEL's loss is functionally identical to OPD Option B, but the teacher signal source is different (experience vs. hint) and there is no sample gating.

### 4.5 Multi-Experience Pool (OEL Consolidate Mode)

In OEL consolidate mode, multiple experience files from different extraction seeds are available. Instead of using a single fixed experience, the system randomly samples from the experience pool for each turn:

```python
def get_experience_for_turn(self) -> str:
    if self._multi_experience:
        with self._experience_pool_lock:
            if self._experience_pool:
                return random.choice(self._experience_pool)
    with self._experience_lock:
        return self._experience_text
```

Experience files are loaded from an `experience_list.txt` (one path per line), built by `tools/make_exp_list.py`:

```
/tmp/oel-extract-round1/global_step_50/extract_100_samples/experiences/experience_50.txt
/tmp/oel-extract-round1/global_step_100/extract_100_samples/experiences/experience_50.txt
/tmp/oel-extract-round1/global_step_150/extract_100_samples/experiences/experience_50.txt
...
```

This diversifies the teacher signal, preventing overfitting to a single experience snapshot.

### 4.6 Trajectory Saving (OEL Deploy Mode)

In deploy mode, each completed session's full trajectory is saved as a JSON file:

```json
{
  "session_id": "session_42",
  "timestamp": "2026-04-09 12:34:56",
  "turns": [
    {
      "turn_num": 1,
      "prompt_ids": [128000, 2675, ...],
      "response_ids": [4546, 9102, ...],
      "response_logprobs": [-0.32, -1.05, ...],
      "prompt_text": "...",
      "response_text": "...",
      "messages": [...]
    },
    ...
  ]
}
```

These trajectories are consumed by the consolidate phase, which reconstructs the student/teacher pairs from the saved data.

### 4.7 Mode-Aware Control Flow

The mode affects behavior at several points:

| Code path | `online` | `extract` | `deploy` | `consolidate` |
|-----------|----------|-----------|----------|----------------|
| Forward to SGLang | Yes | Yes | Yes | Yes |
| Fire teacher task | Yes | **No** | **No** | Yes |
| Submit training sample | Yes | **No** | **No** | Yes |
| Extract experience from session | Yes | Yes | **No** | No |
| Save trajectory to disk | No | Optional | Yes | No |
| Rollout returns samples | Yes | **Empty** | **Empty** | Yes |

In `extract` and `deploy` modes, the rollout function returns an empty `RolloutFnTrainOutput`, and the Slime training loop effectively idles (typically launched with `--val-only` or equivalent flags to prevent errors).

---

## 5. OEL Multi-Round Iteration

The full OEL pipeline iterates across rounds. `run_oel_round.sh` orchestrates:

```
Round 1:
  [Base Model] -> Extract (seeds 50,100,150,200,250) -> experience_list.txt
                -> Deploy -> deploy_data/
                -> Consolidate -> Checkpoint_R1

Round 2:
  [Checkpoint_R1] -> Extract -> experience_list.txt
                   -> Deploy -> deploy_data/
                   -> Consolidate -> Checkpoint_R2

Round N: ...
```

Key OEL paper findings (arXiv:2603.16856) that inform this design:

- **Extracted knowledge >> raw trajectories.** Using self-extracted experience as the teacher augmentation outperforms using raw trajectory text, because extraction filters and generalizes.
- **Self-extracted >> stronger model's knowledge.** Experience extracted by the model itself transfers better than experience extracted by a stronger model, because self-consistency matters more than absolute quality.
- **Multi-round improves monotonically.** Each round accumulates better experience, leading to a better teacher signal, leading to better weights.

---

## 6. Differences from OPD: Source Code Comparison

### 6.1 No Judge, No Voting

OPD's `openclaw_opd_api_server.py` contains:
- `_build_opd_judge_prompt()` -- constructs a judge prompt with `[HINT_START]...[HINT_END]` markers
- `_query_opd_judge_once()` -- calls the PRM engine as a binary judge
- `_parse_hint_text()` -- extracts hint from `[HINT_START]...[HINT_END]` tags
- `_opd_evaluate()` -- runs `m` parallel judge votes, majority-votes, extracts hint

OEL's `openclaw_oel_api_server.py` replaces all of this with:
- `_extract_experience_from_session()` -- extracts experience items from completed sessions
- `_teacher_evaluate()` -- injects experience into prompt, computes teacher logprobs
- No voting logic. No hint parsing. No sample dropping.

### 6.2 Experience State Management

OPD has no persistent state between turns (each hint is independent).

OEL maintains:
- `_experience_text` -- cumulative experience string (thread-safe)
- `_experience_pool` -- list of experience strings for multi-source sampling
- `_session_conversations` -- conversation history for extraction
- `_experience_save_dir` -- persistence to disk
- `_experience_max_tokens` -- FIFO truncation budget

### 6.3 Trajectory Persistence

OPD does not save trajectories. All processing is real-time.

OEL adds `_save_session_trajectory()` for the deploy phase, saving full tokenized trajectories to JSON files for offline consolidation.

### 6.4 Loss Function

Both OPD Option B and OEL use the same top-K reverse KL loss with tail trick. The code is nearly identical:
- OPD: `topk_distillation_loss.py` -> `topk_distillation_loss_function()`
- OEL: `oel_distillation_loss.py` -> `oel_distillation_loss_function()`

The only difference is naming and docstrings.

---

## 7. Environment Variables Reference

### OEL Mode Control

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `OPENCLAW_OEL_MODE` | `online`, `extract`, `deploy`, `consolidate` | `online` | Operating mode |

### Experience Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENCLAW_OEL_EXPERIENCE_PATH` | path | (empty) | Path to initial experience file or `experience_list.txt` |
| `OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH` | int | 2048 | Max experience tokens before FIFO truncation |
| `OPENCLAW_OEL_MULTI_EXPERIENCE` | 0/1 | 0 | Enable random sampling from experience pool |

### Deploy/Trajectory Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENCLAW_OEL_DEPLOY_SAVE_DIR` | path | (empty) | Directory to save session trajectories |

### Inherited from OPCD/OPD

| Variable | Description |
|----------|-------------|
| `SGLANG_API_KEY` | API key for the SGLang chat proxy |
| `HOST` / `PORT` | Bind address for the FastAPI server (default `0.0.0.0:30000`) |
| `SERVED_MODEL_NAME` | Model name for `/v1/chat/completions` (default `qwen3-4b`) |
| `OPENCLAW_RECORD_FILE` | Path to save request/response records |
| `OPENCLAW_RECORD_ENABLED` | 0/1 to enable recording |
| `OPENCLAW_EVAL_MODE` | 0/1 to enable PRM evaluation scoring |
| `PRM_M` | Number of PRM evaluation votes (default 3) |
| `OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY` | Max concurrent teacher logprob queries (default 3) |

---

## 8. Training Configuration (Slime)

Key Slime arguments used across all modes:

```bash
# Loss
--loss-type custom_loss
--custom-loss-function-path oel_distillation_loss.oel_distillation_loss_function
--distill-topk 50
--disable-compute-advantages-and-returns  # no PPO advantages needed
--entropy-coef 0.01                       # optional entropy bonus

# Rollout
--rollout-function-path openclaw_oel_rollout.generate_rollout_openclaw_oel
--custom-generate-function-path openclaw_oel_api_server.generate
--custom-rm-path openclaw_oel_api_server.reward_func

# LoRA (FSDP backend)
--lora-rank 16
--lora-alpha 32
--lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
--lora-modules-to-save embed_tokens lm_head
```

---

## 9. Relationship to openclaw-opcd

`openclaw-oel` is a **strict superset** of `openclaw-opcd`. Running with `OPENCLAW_OEL_MODE=online` and no OEL-specific environment variables produces identical behavior to the original OPCD server. The OPCD module remains as a standalone reference implementation; all new development should use `openclaw-oel`.
