#!/usr/bin/env python3
"""Aggregate a matched multi-harness PawBench matrix."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_agent: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        agent = str(row["agent"])
        task_id = str(row["task_id"])
        if task_id in by_agent[agent]:
            raise ValueError(f"duplicate result: {agent}/{task_id}")
        by_agent[agent][task_id] = row

    agents = sorted(by_agent)
    if len(agents) < 2:
        raise ValueError("at least two harnesses are required")
    task_sets = {agent: set(by_agent[agent]) for agent in agents}
    reference_tasks = task_sets[agents[0]]
    if any(tasks != reference_tasks for tasks in task_sets.values()):
        raise ValueError("harness task sets are not identical")

    harnesses: dict[str, Any] = {}
    for agent in agents:
        arm = list(by_agent[agent].values())
        harnesses[agent] = {
            "tasks": len(arm),
            "full_passes": sum(bool(row["passed"]) for row in arm),
            "mean_score": mean(float(row["score"]) for row in arm),
            "status_failures": sum(row["status"] != "success" for row in arm),
            "api_error_anomalies": sum(
                bool((row.get("anomaly") or {}).get("has_api_error")) for row in arm
            ),
            "anomalous_tasks": sorted(
                row["task_id"]
                for row in arm
                if bool((row.get("anomaly") or {}).get("is_anomalous"))
            ),
            "total_execution_seconds": sum(float(row["execution_time"]) for row in arm),
        }

    pairs: dict[str, Any] = {}
    for left, right in itertools.combinations(agents, 2):
        left_only = sorted(
            task for task in reference_tasks
            if by_agent[left][task]["passed"] and not by_agent[right][task]["passed"]
        )
        right_only = sorted(
            task for task in reference_tasks
            if by_agent[right][task]["passed"] and not by_agent[left][task]["passed"]
        )
        score_deltas = {
            task: float(by_agent[left][task]["score"])
            - float(by_agent[right][task]["score"])
            for task in reference_tasks
        }
        pairs[f"{left}_vs_{right}"] = {
            "left": left,
            "right": right,
            "left_pass_right_fail": left_only,
            "right_pass_left_fail": right_only,
            "paired_net_full_passes": len(left_only) - len(right_only),
            "mean_score_delta": mean(score_deltas.values()),
            "largest_absolute_score_gaps": [
                {"task_id": task, "score_delta": score_deltas[task]}
                for task in sorted(
                    score_deltas,
                    key=lambda task: (-abs(score_deltas[task]), task),
                )[:5]
            ],
        }

    return {
        "agents": agents,
        "task_count": len(reference_tasks),
        "harnesses": harnesses,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    result = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
