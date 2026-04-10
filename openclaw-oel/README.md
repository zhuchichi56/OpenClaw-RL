# Online Experiential Learning with Context Distillation

Online experiential learning for agentic tool-use: accumulate transferable experiential knowledge from deployment trajectories, then consolidate it into model parameters via on-policy context distillation — no reward signal required.

## Method Overview

### Online Mode (Single-Process Continuous Training)

The policy model is deployed as an OpenAI-compatible chat proxy. External environments (e.g. OpenClaw) send multi-turn conversations through this proxy. The system runs **experience accumulation** and **context distillation** simultaneously:

1. Forward each request to the policy model (SGLang) and collect the response with per-token log-probabilities.
2. When a **session completes**, extract structured experience items from the full conversation via the PRM/teacher engine.
3. Experience items are appended to a global experience buffer (FIFO, truncated to `max_tokens`).
4. For each **main-line turn**, inject the accumulated experience into the last user message to form the teacher prompt:

   $$\text{teacher input} = [\text{experience};\, x], \qquad \text{student input} = x$$

5. Query teacher log-probs on the original response tokens.
6. Submit training sample (student log-probs + teacher log-probs) to SLIME.

### OEL Iterative Mode (Multi-Round 3-Phase Pipeline)

Following [OEL (arXiv 2603.16856)](https://arxiv.org/abs/2603.16856), training is split into explicit phases per round:

1. **Extract** — Deploy model, collect interaction trajectories, extract experiential knowledge into text files. Inference only; no weight updates.
2. **Deploy** — Collect deployment trajectories (prompt + response + logprobs) and save to disk. Inference only.
3. **Consolidate** — Load experience files + saved trajectories. Construct teacher (model + experience) and student (model alone). Train with reverse-KL distillation.

Rounds iterate: the consolidated checkpoint becomes the starting model for the next round's extract phase.

## Distillation Loss

### Reverse KL with Top-K Approximation

For each response token position $t$, minimize the reverse KL divergence between student and teacher over the teacher's top-$K$ vocabulary tokens plus a tail bin:

$$\mathcal{L}=\frac{1}{|y|}\sum_{t=1}^{|y|}D_{\text{KL}}\!\left(\pi_\theta(\cdot\mid x,y_{<t})\;\big\|\;\pi_{\text{teacher}}(\cdot\mid e,x,y_{<t})\right)$$

Top-$K$ approximation with tail mass trick:

$$D_{\text{KL}}^{K+1}=\sum_{k=1}^{K+1}\pi_\theta^{(k)}\!\left(\log\pi_\theta^{(k)}-\log\pi_{\text{teacher}}^{(k)}\right)$$

where the $(K+1)$-th bin captures the remaining probability mass:

$$\log p_{\text{tail}}=\log\!\left(1-\exp\!\left(\text{logsumexp}(\log p_1,\dots,\log p_K)\right)\right)$$

Default: $K=50$, with optional entropy bonus ($\beta_{\text{ent}}=0.01$).

### Why Reverse KL

| Property | Reverse KL (ours) | Forward KL (standard CD) |
|----------|--------------------|--------------------------|
| Sampling | On-policy (student generates) | Off-policy (teacher generates) |
| Behavior | Mode-seeking | Mode-covering |
| OOD preservation | Strong | Degrades |

## How to Run

### Online Mode

```bash
cd slime
bash ../openclaw-oel/run_qwen3_4b_openclaw_oel_online.sh
```

### OEL Iterative Mode

```bash
cd slime
# Full round (extract → deploy → consolidate)
ROUND=1 MODEL_PATH=../models/Qwen3-4B bash ../openclaw-oel/run_oel_round.sh

# Or run phases individually:
# Phase 1: Extract
EXP_NAME=oel-extract-round1 SEED=42 bash ../openclaw-oel/run_qwen3_4b_openclaw_oel_extract.sh
# Phase 2: Deploy
EXP_NAME=oel-deploy-round1 bash ../openclaw-oel/run_qwen3_4b_openclaw_oel_deploy.sh
# Phase 3: Consolidate
EXP_NAME=oel-consolidate-round1 \
  EXP_PATH=/tmp/oel-extract-round1/experience_list.txt \
  bash ../openclaw-oel/run_qwen3_4b_openclaw_oel_consolidate.sh
```

## File Layout

```text
openclaw-oel/
├── README.md
├── openclaw_oel_api_server.py               # FastAPI proxy + experience accumulation + teacher logprobs
├── openclaw_oel_rollout.py                  # Async rollout worker (bridges API server ↔ SLIME trainer)
├── oel_distillation_loss.py                 # Reverse-KL top-K loss (external custom loss)
├── run_qwen3_4b_openclaw_oel_online.sh      # Online mode (continuous training)
├── run_qwen3_4b_openclaw_oel_extract.sh     # OEL Phase 1: experience extraction
├── run_qwen3_4b_openclaw_oel_deploy.sh      # OEL Phase 2: trajectory collection
├── run_qwen3_4b_openclaw_oel_consolidate.sh # OEL Phase 3: consolidation training
├── run_oel_round.sh                         # Multi-round iteration wrapper
├── docs/
│   └── DESIGN.md                            # Architecture, OEL vs OPD comparison, impl details
├── tools/
│   └── make_exp_list.py                     # Build experience file list for consolidation
└── results/                                 # Runtime records (auto-created)
```

## References

- [Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) (OEL)
- [On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) (OPCD)
