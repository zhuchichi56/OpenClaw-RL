# Development Log

## 2026-08-03 — Workplace harness-adaptation Phase 0

- **Question:** Does changing QwenPaw, OpenClaw, or Hermes while holding the model, task, endpoint, decoding budget, and grader fixed create a measurable workplace-performance gap?
- **Decision:** The frozen 24×3 PawBench matrix passes the Phase 0 gate. QwenPaw has 8/24 full passes versus 5/24 for OpenClaw and Hermes; both paired net differences are 3, with at least four non-timeout harness-attributable traces.
- **Solution:** Added a frozen manifest, shared request-clamp proxy, reproducible PawBench overlay/runner, automated-only hybrid grading, deterministic aggregation, tests, and a decision report. No parameter update or remote GPU work was run.
- **Owner and artifacts:** This branch owns `plan.md`, `scripts/harness_adaptation/`, `tests/harness_adaptation/`, and `experiment/harness_adaptation/`. Full raw artifacts are sealed at the path and hash in `experiment/harness_adaptation/artifact_manifest.json`.
- **Verification:** 72/72 formal executions produced deterministic results; 0 API/status failures; 7/7 local tests pass including the real PawBench overlay integration test; archive size/hash/readability verified.
- **Status:** Phase 0 complete. Next is the non-training WorkBench substrate parity gate; training remains unauthorized.
- **Commit:** pending
