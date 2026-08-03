# Workplace Harness Continual Adaptation

> Status: Phase 0 complete and passed. The earlier Qwen Code to Codex comparison is
> preserved as completed preliminary evidence through Wave 1, but Wave 2/3 are
> paused because both runtimes are coding-oriented and do not provide a strong
> test of continual harness adaptation. The PawBench causal pilot passed its
> frozen gap gate; no parameter update is authorized until the Phase 1
> WorkBench substrate gate also passes.

## Research Question

Holding model weights, task, workspace, sampling budget, and evaluator fixed,
how much does changing the agent harness alter realized workplace performance?
If a measurable harness gap exists, can sequential experience distillation
adapt the same weights to new harnesses while retaining performance on earlier
harnesses?

The harness is the treatment. It includes its system prompt, context and memory
policy, tool protocol, workspace conventions, control loop, recovery behavior,
and termination rules. A model checkpoint or benchmark is not a harness.

## Phase 0: Causal Harness-Gap Pilot

Use PawBench because it already exposes the same task and workspace through
three structurally different harnesses: QwenPaw, OpenClaw, and Hermes.

| Item | Frozen pilot rule |
|---|---|
| Model | One frozen Qwen3.5-4B checkpoint and revision for all arms |
| Harnesses | QwenPaw, OpenClaw, Hermes at pinned container/runtime versions |
| Tasks | 24 closed-environment, text-only workplace tasks; freeze IDs and dataset commit before execution |
| Grading | Deterministic automated checks only for the primary gate; LLM-judge scores are diagnostic |
| Budget | Same model endpoint, context limit, maximum generated tokens, wall-clock limit, and task files |
| Repetition | One infrastructure smoke task, then the frozen 24-task matrix; repeat only the decision-critical pair if stochasticity is material |
| Artifacts | Per-task transcript, workspace, model/harness identities, request counts, latency, deterministic score, and anomaly status |

The pilot passes only if at least one harness pair has both:

1. a paired net difference of at least 3 of 24 tasks on deterministic success;
2. at least three task failures attributable from traces to harness behavior,
   rather than model-server, grader, timeout, or broken-environment errors.

If the gate fails, do not train. Expand once to a 60-task deterministic slice
or reject this harness set. Do not tune prompts or harness settings on the
frozen pilot tasks.

## Phase 1: Training Substrate Gate

Only after Phase 0 passes, adapt WorkBench's 690 workplace tasks to the selected
harnesses. WorkBench contributes the business state, 26 read/write tools, and
deterministic final-state and harmful-side-effect evaluators; it does not become
a fourth harness.

Partition by `base_template`, not by random task rows, so paraphrases of one
template cannot cross train/dev/test boundaries. Freeze the exact task IDs,
template groups, source commit, tool schemas, initial database hashes, and
evaluator version. Before collecting training data, require one task per domain
to pass state reset, tool execution, transcript capture, final-state grading,
and harmful-side-effect grading under every selected harness.

## Phase 2: Continual Adaptation

Run the same checkpoint sequentially through harness waves selected from the
Phase 0 gap. Each wave uses disjoint WorkBench training templates. After every
wave, evaluate all seen and unseen harnesses on the frozen dev/test templates.

```text
M0 -> H1 adaptation -> M1 -> H2 adaptation -> M2 -> H3 adaptation -> M3
      evaluate H1/H2/H3 after every checkpoint
```

Reuse only validated components:

- PawBench normalized transcripts and workspace artifacts for execution;
- the existing OEL/OPCD teacher-replay and Qwen3.5 training backend after a
  real train/save/reload/eval smoke;
- Polar's sealed-target identity and integrity checks where their assumptions
  remain valid.

Do not reuse the coding-specific global experience, SWE tasks, Codex/Qwen Code
prompts, or SWE-bench evaluator as workplace evidence.

Primary metrics are current-harness gain, forward transfer, backward transfer
or forgetting, average accuracy across harnesses, harmful side effects, and
paired task transitions. Report tool-selection, state-tracking, recovery, and
termination failures separately from model-server and environment failures.

## External Test and Stop Rules

Keep WildClawBench entirely outside prompt, adapter, experience, and checkpoint
selection. Use it once after the continual sequence as an external work-scenario
test.

Stop before training if the Phase 0 gap is absent, WorkBench state/evaluator
parity fails across harnesses, task-template leakage is found, or a real
save/reload/evaluation smoke fails. A negative result after a valid continual
sequence is still a result and must not be hidden or replaced by a selected
intermediate checkpoint.

## Immediate Next Action

Implement the Phase 1 WorkBench substrate gate without training: freeze the
WorkBench commit, template-group split, tool schemas, initial state hashes, and
evaluators; then require one task per domain to pass state reset, tool execution,
transcript capture, final-state grading, and harmful-side-effect grading under
QwenPaw, OpenClaw, and Hermes. Hermes must first pass a runtime canary for the
timeout failure observed in Phase 0. The verified Phase 0 results are in
`experiment/harness_adaptation/P0_REPORT.md`.
