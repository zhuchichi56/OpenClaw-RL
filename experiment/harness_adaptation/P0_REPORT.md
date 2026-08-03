# Phase 0 Workplace Harness-Gap Pilot

## Decision snapshot

| Item | Verdict | Evidence boundary |
|---|---|---|
| Question | Does changing only the workplace agent harness materially change realized task performance? | Same frozen model, 24 paired tasks, shared endpoint and deterministic checks |
| Run status | **Complete: 72/72 task-harness executions** | `P0-matrix-24x3-final`; one execution per paired cell |
| Primary gate | **Passed** | QwenPaw exceeds both OpenClaw and Hermes by a paired net 3 full passes; at least four non-timeout trace differences are attributable to harness behavior |
| Supported conclusion | The same 4B model is sensitive enough to workplace harness semantics to justify a harness-adaptation substrate study. | Supported for this frozen PawBench slice; not yet a training result or a robustness claim |
| Next decision | Run the WorkBench substrate parity gate before collecting training data. | No parameter update is authorized by Phase 0 |

## Frozen protocol

The treatment is the harness: QwenPaw, OpenClaw, or Hermes. All arms used
`custom/qwen3.5-4b-polar` at revision
`568507d78f31db25ab0abeb97177b86186798a19`, the same OpenAI-compatible
endpoint, temperature 0, 65,536-token context, and 8,192 maximum generated
tokens. The 24 closed, text-only tasks have frozen identity
`aae32b78a19c11300120fc6a2976246ec3fd0fde1de24b4667193d1c4e5448d5` at
PawBench commit `0f794a8bb6c27aa9ee4091b2691fa30e4ed9cc8f`. Primary scores use only the
embedded deterministic checks; no LLM judge contributes to the result.

Exact identities and budgets are in
[`pawbench_pilot_manifest.json`](pawbench_pilot_manifest.json). The smoke run
is operational evidence only and is excluded from every number below.

## Main result

| Harness | Completed | Full pass | Mean deterministic score | Total execution time | API/status failures |
|---|---:|---:|---:|---:|---:|
| QwenPaw | 24/24 | **8/24** | **0.8075** | 2,214.7 s | 0 |
| OpenClaw | 24/24 | 5/24 | 0.7387 | 2,100.0 s | 0 |
| Hermes | 24/24 | 5/24 | 0.7003 | 2,958.7 s | 0 |

`Full pass` means the deterministic score is exactly 1.0. The comparison is
paired by task. It has one run per cell, so the table is a verified result for
the frozen matrix but does not estimate run-to-run variance.

| Pair (left minus right) | Left-only full passes | Right-only full passes | Paired net | Mean-score delta | Gate |
|---|---:|---:|---:|---:|---|
| QwenPaw − OpenClaw | 5 | 2 | **+3** | **+0.0688** | Pass |
| QwenPaw − Hermes | 4 | 1 | **+3** | **+0.1072** | Pass |
| OpenClaw − Hermes | 4 | 4 | 0 | +0.0384 | No net gap |

The exact transition IDs and score deltas are machine-readable in
[`analysis.json`](runs/P0-matrix-24x3-final/analysis.json). Raw native results
are under `runs/P0-matrix-24x3-final/{qwenpaw,openclaw,hermes}/`.

## Trace attribution

The following cases satisfy the attribution requirement without relying on an
LLM judge, a model-server error, or a broken grader. Each comparison uses the
same task and model; the differing action sequence and terminal workspace are
visible in the linked transcripts.

| Task | Deterministic scores | Harness behavior and terminal evidence | Attribution |
|---|---|---|---|
| `task_openclaw_comprehension` | QwenPaw 0.000; OpenClaw 0.833; Hermes 0.000 | QwenPaw exhausted browser and missing PDF-library paths without writing `answer.txt`. Hermes issued one `read_file` call and stopped. OpenClaw created a venv, installed a PDF reader, extracted the document, and wrote `answer.txt`. | Tool availability, recovery policy, and termination differ by harness. |
| `task_00052_generate_openai_social_media_profile_from_workspace_data` | QwenPaw 0.812; OpenClaw 0.000; Hermes 0.781 | QwenPaw and Hermes repeatedly wrote `output/openai_social_profile.json`. OpenClaw read a subset of inputs, searched memory/files, then terminated without the required JSON. | State discovery and completion/termination behavior differ. |
| `task_00100_house_robber_algorithm_deep_dive_explanation` | QwenPaw 0.867; OpenClaw 0.000; Hermes 0.833 | All three read the reference files. QwenPaw and Hermes created both the explanation and verification JSON; OpenClaw stopped after reading and created neither. | Terminal-action policy, not missing task evidence, explains the collapse. |
| `T003zh_calendar_scheduling` | QwenPaw 1.000; OpenClaw 1.000; Hermes 0.750 | QwenPaw/OpenClaw wrote `output/scheduled_event.json` relative to the workspace. Hermes wrote `/workspace/output/scheduled_event.json`; the grader could not find the required workspace-relative artifact. | Harness-specific workspace path semantics caused the loss. |
| `CTB_A02_investment_priority_matrix` | QwenPaw 0.778; OpenClaw 1.000; Hermes 0.000 | QwenPaw computed and wrote a report but omitted two composite scores. OpenClaw completed all scores. Hermes produced a `KeyboardInterrupt` traceback after 250 s and no report. | QwenPaw/OpenClaw difference is attributable; the Hermes arm is excluded from the non-timeout trace count. |

The first four rows are sufficient for the frozen `>=3` attributable-trace
gate. The shared 0 on `task_00075_sector_momentum_rotation_backtest_with_data_quality_traps`
is not counted: all harnesses failed, and Hermes reached the 900-second limit.
Hermes `CTB_A02` and `CTB_A03` are also excluded from the trace gate because
their terminal `KeyboardInterrupt` coincides with the task time limit.

## Failure-layer audit

| Layer | Observed evidence | Decision |
|---|---|---|
| Model server | 0/72 API-error anomalies; all rows have `status=success` | Does not explain the gap |
| Grader | 72/72 deterministic breakdowns; hybrid tasks bypassed the LLM judge | Does not explain the gap |
| Environment/archive | Six rejected optional log-file copies, exactly two per harness; required workspaces and graders remained readable | Archive warnings only; no scored task invalidated |
| Harness/runtime | Seven `SHORT_TRANSCRIPT` flags: the three `task_blog` flags still produced 0.90–0.95 and are heuristic false positives; four Hermes flags coincide with real terminal failures | Preserve separately from task-quality errors |
| Common task difficulty | Sector-backtest scored 0 in all three arms | Not harness-discriminating |

## Claim status and next gate

| Claim | Verdict | Boundary |
|---|---|---|
| Harness choice changes workplace performance for a frozen model. | **Supported** | Two pairs meet the preregistered +3/24 full-pass gate, with non-timeout trace evidence. |
| One harness is uniformly superior. | **Unsupported** | Transitions are bidirectional; OpenClaw uniquely succeeds on PDF comprehension while failing other terminal-write tasks. |
| The observed gap is robust across seeds, task samples, or models. | **Open** | One deterministic run per task and one 24-task slice. |
| Continual harness adaptation improves the model. | **Open** | No training has been run. |

Phase 0 therefore authorizes only Phase 1: adapt WorkBench execution to the
three harnesses and require one task per domain to pass reset, tool execution,
transcript capture, final-state grading, and harmful-side-effect grading.
Hermes must additionally pass a runtime canary that distinguishes harness
process timeout from task failure. Stop before training if state/evaluator
parity fails or template leakage is found.

## Reproducibility

- Matrix summary: [`run_summary.json`](runs/P0-matrix-24x3-final/run_summary.json)
- Deterministic aggregation: [`analysis.json`](runs/P0-matrix-24x3-final/analysis.json)
- Full runner log: [`runner.log`](runs/P0-matrix-24x3-final/runner.log)
- Request clamp audit: [`requests.jsonl`](server_audit/requests.jsonl)
- Generator and runner: [`pawbench_pilot_manifest.py`](../../scripts/harness_adaptation/pawbench_pilot_manifest.py), [`pawbench_pilot_runner.py`](../../scripts/harness_adaptation/pawbench_pilot_runner.py)
- Durable raw-artifact bundle: [`artifact_manifest.json`](artifact_manifest.json), verified by equal local/Blob SHA-256 and readable 5,405-file archive inventory
