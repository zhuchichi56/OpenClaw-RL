# Development Log

## 2026-08-09 — PawBench 24×3 trajectory mechanism analysis

- **Question:** Why does QwenPaw lead the frozen 24-task matrix, where do OpenClaw and Hermes win, and which trajectory mechanisms are reusable for Polar experience learning?
- **Decision:** QwenPaw's aggregate lead is concentrated in the four QwenClawBench long-horizon tasks and is associated with artifact-producing persistence, not uniform superiority. OpenClaw is strongest on concise completion and tiered recovery; Hermes is competitive on direct file/compute tasks but loses to workspace-path and runtime failures. The comparison is not compute-matched.
- **Solution:** Added a reproducible 72-trajectory analyzer, machine-readable statistics, a full mechanism report, a 24-task score/turn ledger, and evidence-bounded contrastive experience rules.
- **Owner and artifacts:** `scripts/harness_adaptation/analyze_pawbench_trajectories.py`, `experiment/harness_adaptation/runs/P0-matrix-24x3-final/trajectory_analysis.json`, and `experiment/harness_adaptation/P0_TRAJECTORY_ANALYSIS.md`.
- **Verification:** Analyzer re-read all 72 transcripts and 72 native results; task sets match; output reports 24 tasks/72 trajectories; JSON parses successfully; report values were checked against the generated artifact.
- **Status:** Analysis complete. The experience rules are proposals with trajectory-level evidence and still require held-out, no-experience-prompt internalization tests.
- **Commit:** `6ea9aee`

## 2026-08-03 — Workplace harness-adaptation Phase 0

- **Question:** Does changing QwenPaw, OpenClaw, or Hermes while holding the model, task, endpoint, decoding budget, and grader fixed create a measurable workplace-performance gap?
- **Decision:** The frozen 24×3 PawBench matrix passes the Phase 0 gate. QwenPaw has 8/24 full passes versus 5/24 for OpenClaw and Hermes; both paired net differences are 3, with at least four non-timeout harness-attributable traces.
- **Solution:** Added a frozen manifest, shared request-clamp proxy, reproducible PawBench overlay/runner, automated-only hybrid grading, deterministic aggregation, tests, and a decision report. No parameter update or remote GPU work was run.
- **Owner and artifacts:** This branch owns `plan.md`, `scripts/harness_adaptation/`, `tests/harness_adaptation/`, and `experiment/harness_adaptation/`. Full raw artifacts are sealed at the path and hash in `experiment/harness_adaptation/artifact_manifest.json`.
- **Verification:** 72/72 formal executions produced deterministic results; 0 API/status failures; 7/7 local tests pass including the real PawBench overlay integration test; archive size/hash/readability verified.
- **Status:** Phase 0 complete. Next is the non-training WorkBench substrate parity gate; training remains unauthorized.
- **Commit:** `5aca1ec`
