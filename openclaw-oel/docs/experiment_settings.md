# Experiment Settings: OPCD/OEL-style vs SDFT/SDPO-style

## Task: Personal Agent (GSM8K Math Personalization)

The model learns to personalize its writing style based on user feedback during
multi-turn conversations, evaluated on GSM8K math problems.

### Two Scenarios (alternating per round)

**Scenario A — Lazy Student** (even rounds):
- Simulator (GPT-4.1) plays a lazy student who wants homework solved
- Goal: AI produces a full math solution in **natural, human-like style** — no markdown,
  no bold, no numbered lists, no "Final answer:" patterns
- Student says "DONE" when satisfied with the style

**Scenario B — Teacher Grading** (odd rounds):
- Simulator (GPT-4.1) plays a math teacher grading student homework
- Goal: AI produces warm, specific, encouraging feedback in **natural prose** — no bullet
  points, no headers, no AI formatting
- Teacher says "DONE" when satisfied

Both scenarios test the same skill: **can the model produce non-AI-looking responses
while maintaining substance?**

## Model & Training

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-1.7B |
| Training mode | Full-parameter (no LoRA) |
| Backend | Megatron (tensor-parallel) |
| GPUs | 4x NVIDIA H100 80GB |
| GPU layout | actor=2, rollout=1, PRM/teacher=1 |
| Learning rate | 1e-5 (constant) |
| Optimizer | Adam (beta1=0.9, beta2=0.98, weight_decay=0.1) |
| Batch size | 16 samples per weight update |
| Loss | KL(student \|\| teacher) with top-K=50 + tail approximation |
| Entropy coef | 0.0 |
| Training rounds | 40 |
| Sessions per round | 2 |
| Total training sessions | 80 |
| Max turns per session | 6 |
| Policy temperature (train) | 0.6 |
| Policy max tokens | 8192 (rollout), 2048 (eval) |

## Data: Hard Problem Selection

### Selection Process

1. Evaluate 180 GSM8K problems with the **untrained** Qwen3-1.7B model (Student scenario)
2. Score each with GPT-4.1 evaluator, 5-vote majority
3. Keep problems where **baseline score <= 0.25** (model performs poorly)
4. Result: 72 hard problems found
5. Split 50/50 into train (36) and eval (36) sets

### Data Files

| File | Count | Purpose |
|------|-------|---------|
| `hard_problems_train.json` | 36 | Training sessions draw from these |
| `hard_problems_eval.json` | 36 | Evaluation uses these (no overlap) |

Each problem has: `idx`, `question`, `ground_truth_answer`, `baseline_score`

Baseline scores range from 0.05 to 0.25.

## Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Evaluator model | GPT-4o (Azure OpenAI) |
| Evaluator temperature | 0.0 |
| Voting | 5 votes per (question, response) pair |
| Score scale | {0, 0.25, 0.5, 0.75, 1.0} |
| Final score | Average of all successful votes |
| Student eval problems | 36 |
| Teacher eval problems | 24 |
| Eval frequency | Every 2 weight updates |
| Policy temperature (eval) | 0.3 |

### Evaluation Flow

1. Send one question to policy model (`turn_type="side"`, no training signal)
2. Strip `<think>...</think>` blocks from response
3. GPT-4o scores the response 5 times independently
4. Each vote extracts score from `\boxed{}` in evaluator output
5. Average all valid votes → final score for that problem
6. Report mean across all problems

### What the Evaluator Scores

The evaluator checks whether the response satisfies the **PREFERENCE** (personalization
quality), NOT math correctness. Specifically:
- Natural, casual language (no AI-like formatting)
- Full reasoning shown but in human-like writing style
- No markdown artifacts (bold, headers, numbered lists)

## Simulator

| Parameter | Value |
|-----------|-------|
| Simulator model | GPT-4.1 (Azure OpenAI) |
| Simulator temperature | 0.0 |
| Role | Plays as the user (student or teacher persona) |
| Feedback | Provides natural language feedback on style quality |
| Termination | Says "DONE" when satisfied, or max 6 turns |

## Methods Compared

### OPCD/OEL-style (Ours)

Based on OPCD (On-Policy Context Distillation, arXiv 2602.12275) and OEL (Online
Experiential Learning, arXiv 2603.16856).

**Core mechanism**: Experience-augmented context distillation
1. Student model serves user with bare prompts (no experience)
2. Teacher model (same architecture) sees prompts augmented with **accumulated experience**
3. Student trained to match teacher via KL divergence (top-K + tail)
4. After each session, PRM extracts structured experience items from the conversation
5. Experience accumulates across sessions (capped at 2048 tokens)

**Key properties**:
- **Persistent experience**: grows across sessions, saved to disk
- **No sample dropping**: all turns are training signal (100% sample utilization)
- **Last turn recovered**: uses `[session ended]` as synthetic next-state
- **Experience injection**: wraps user prompt with experience template

**Experience extraction prompt (v2)**: demands concrete do/don't rules with specific
examples, supports deduplication ("No new experience." if nothing novel).

### SDFT/SDPO-style

Based on SDFT (Self-Distillation Fine-Tuning) and SDPO (Self-Distillation with
Preference Optimization).

**Core mechanism**: Per-turn hindsight hint distillation
1. Student model serves user with bare prompts
2. When next turn arrives, a **judge** (PRM) evaluates whether hindsight is available
3. Judge votes (m=3): `\boxed{1}` (useful hint exists) or `\boxed{-1}` (no hint)
4. If accepted: longest valid hint is extracted from `[HINT_START]...[HINT_END]`
5. Teacher sees prompt with hint appended: `[user's hint / instruction]\n{hint}`
6. Student trained to match teacher via KL divergence (top-K + tail)

**Key properties**:
- **Ephemeral hints**: per-turn, discarded after use, not accumulated
- **Sample dropping**: turns without valid hints are discarded (~30-50% rejection rate)
- **Last turn dropped**: no next-state → no hint → rejected
- **Hint injection**: appends hint text to last user message

### Algorithmic Differences Summary

| Dimension | OPCD/OEL-style (Ours) | SDFT/SDPO-style |
|-----------|-----------------------|-----------------|
| Teacher context | Accumulated experience (session-level) | Per-turn hindsight hint |
| Context lifecycle | Persistent, cross-session | Ephemeral, per-turn |
| Sample acceptance | All accepted (100%) | Filtered by judge (~50-70%) |
| Last turn | Processed (synthetic next-state) | Dropped |
| Compute per turn | 1 teacher logprob query | m judge queries + 1 teacher logprob |
| Information type | General strategies | Instance-specific hindsight |
| Loss function | KL(student \|\| teacher), top-K=50 | KL(student \|\| teacher), top-K=50 |

**Loss function is identical** between the two methods — same KL divergence with top-K
approximation and tail trick. The only difference is *what privileged context the teacher
receives*.

## Results

| Method | Baseline | Peak | Final | Weight Updates |
|--------|----------|------|-------|----------------|
| **OPCD/OEL-style (Ours)** | 0.315 | **0.460** | **0.460** | 23 |
| SDFT/SDPO-style | 0.304 | 0.390 | 0.353 | 12 |

![Student Score vs Round](../scripts/student_score_opcd_vs_sdft.png)

OPCD/OEL-style achieves +46% relative improvement over baseline (0.315 → 0.460),
while SDFT/SDPO-style peaks at +28% (0.304 → 0.390) but regresses to +16% (0.353)
by the end.

## Reproduction

```bash
# Start training backend (inside openclaw-slime container)
docker exec -d openclaw-slime bash -c "
  cd /workspace/OpenClaw-RL/openclaw-oel
  bash run_qwen3_1.7b_openclaw_oel_online.sh > /tmp/oel_train.log 2>&1
"

# Wait ~2 min for sglang ready, then run experiment
cd /home/v-hezhu2/OpenClaw-RL/openclaw-test-new
python3 -u gsm8k_personal_agent.py \
  --method oel \
  --training-rounds 40 \
  --eval-every 2 \
  --problem-file results/hard_problems_train.json \
  --eval-problem-file results/hard_problems_eval.json \
  --evaluator-model gpt-4o
```
