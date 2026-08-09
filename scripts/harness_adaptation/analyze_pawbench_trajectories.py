#!/usr/bin/env python3
"""Analyze the frozen PawBench 24-task, three-harness trajectory matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HARNESSES = ("qwenpaw", "openclaw", "hermes")
READ_TOOLS = {"read", "read_file", "search_files", "glob_search", "memory_search"}
WRITE_TOOLS = {"write", "write_file", "edit", "edit_file", "patch", "send_file_to_user"}
SHELL_TOOLS = {"exec", "execute_shell_command", "terminal", "execute_code"}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def result_file(harness_dir: Path) -> Path:
    candidates = sorted(harness_dir.glob("20*.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one native result in {harness_dir}, got {candidates}")
    return candidates[0]


def transcript_stats(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assistant_turns = 0
    calls: list[dict[str, Any]] = []
    final_texts: list[str] = []
    for row in rows:
        message = row.get("message") or {}
        if message.get("role") != "assistant":
            continue
        assistant_turns += 1
        turn_calls = 0
        text_parts: list[str] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "toolCall":
                turn_calls += 1
                calls.append({"name": block.get("name", ""), "arguments": block.get("arguments") or {}})
            elif block.get("type") == "text" and str(block.get("text", "")).strip():
                text_parts.append(str(block["text"]))
        if not turn_calls and text_parts:
            final_texts.append("\n".join(text_parts))

    names = [call["name"] for call in calls]
    call_keys = [
        json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for call in calls
    ]
    repeated_calls = sum(1 for left, right in zip(call_keys, call_keys[1:]) if left == right)
    return {
        "events": len(rows),
        "assistant_turns": assistant_turns,
        "tool_calls": len(calls),
        "tool_names": dict(sorted(Counter(names).items())),
        "read_calls": sum(name in READ_TOOLS for name in names),
        "write_calls": sum(name in WRITE_TOOLS for name in names),
        "shell_calls": sum(name in SHELL_TOOLS for name in names),
        "used_read": any(name in READ_TOOLS for name in names),
        "used_write": any(name in WRITE_TOOLS for name in names),
        "used_shell": any(name in SHELL_TOOLS for name in names),
        "consecutive_exact_repeats": repeated_calls,
        "has_final_text": bool(final_texts),
    }


def prompt_metadata(run_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "qwenpaw": {"system_prompt_chars": None, "declared_tool_count": None},
        "openclaw": {"system_prompt_chars": None, "declared_tool_count": None},
        "hermes": {"system_prompt_chars": None, "declared_tool_count": None},
    }
    qwen_session = next(
        (run_root / "qwenpaw" / "workspaces" / "T002_email_triage" / "sessions").glob("*.json")
    )
    qwen = load_json(qwen_session)
    for request in (qwen.get("agent") or {}).get("_model_trajectory") or []:
        for message in request.get("messages") or []:
            if message.get("role") == "system":
                content = message.get("content")
                rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                metadata["qwenpaw"]["system_prompt_chars"] = len(rendered)
                break
        if metadata["qwenpaw"]["system_prompt_chars"] is not None:
            break

    trajectory = next(
        (run_root / "openclaw" / "workspaces" / "T002_email_triage" / "sessions").glob(
            "*.trajectory.jsonl"
        )
    )
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") == "session.started":
            metadata["openclaw"]["declared_tool_count"] = (event.get("data") or {}).get("toolCount")
        if event.get("type") == "trace.metadata":
            report = (((event.get("data") or {}).get("prompting") or {}).get("systemPromptReport") or {})
            metadata["openclaw"]["system_prompt_chars"] = (report.get("systemPrompt") or {}).get("chars")
    return metadata


def task_source(filename: str) -> str:
    lowered = filename.lower()
    for source in ("claweval", "qwenclawbench", "pinchbench"):
        if source in lowered:
            return source
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    task_meta = {task["task_id"]: task for task in manifest["tasks"]}
    native: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for harness in HARNESSES:
        payload = load_json(result_file(args.run_root / harness))
        payloads[harness] = payload
        native[harness] = {row["task_id"]: row for row in payload["results"]}
        for path in sorted((args.run_root / harness / "transcripts").glob("*.jsonl")):
            traces[harness][path.stem] = transcript_stats(path)

    reference = set(native[HARNESSES[0]])
    if any(set(native[harness]) != reference for harness in HARNESSES):
        raise ValueError("harness task sets differ")
    if any(set(traces[harness]) != reference for harness in HARNESSES):
        raise ValueError("transcript task sets differ")

    harness_summary: dict[str, Any] = {}
    for harness in HARNESSES:
        rows = native[harness]
        task_traces = traces[harness]
        tools = Counter()
        for trace in task_traces.values():
            tools.update(trace["tool_names"])
        harness_summary[harness] = {
            "tasks": len(rows),
            "full_passes": sum(math.isclose(row["score"], 1.0) for row in rows.values()),
            "mean_score": statistics.mean(row["score"] for row in rows.values()),
            "estimated_usage": payloads[harness]["summary"].get("total_usage", {}),
            "execution_seconds": quantiles([row["execution_time"] for row in rows.values()]),
            "assistant_turns": quantiles([trace["assistant_turns"] for trace in task_traces.values()]),
            "tool_calls": quantiles([trace["tool_calls"] for trace in task_traces.values()]),
            "tool_name_counts": dict(sorted(tools.items())),
            "read_task_coverage": sum(trace["used_read"] for trace in task_traces.values()),
            "write_task_coverage": sum(trace["used_write"] for trace in task_traces.values()),
            "shell_task_coverage": sum(trace["used_shell"] for trace in task_traces.values()),
            "zero_tool_tasks": sum(trace["tool_calls"] == 0 for trace in task_traces.values()),
            "total_consecutive_exact_repeats": sum(
                trace["consecutive_exact_repeats"] for trace in task_traces.values()
            ),
        }

    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(HARNESSES):
        for right in HARNESSES[left_index + 1 :]:
            deltas = {task: native[left][task]["score"] - native[right][task]["score"] for task in reference}
            pairwise[f"{left}_vs_{right}"] = {
                "left_score_wins": sum(delta > 1e-12 for delta in deltas.values()),
                "ties": sum(abs(delta) <= 1e-12 for delta in deltas.values()),
                "right_score_wins": sum(delta < -1e-12 for delta in deltas.values()),
                "mean_score_delta": statistics.mean(deltas.values()),
                "left_full_pass_right_not": sum(
                    math.isclose(native[left][task]["score"], 1.0)
                    and not math.isclose(native[right][task]["score"], 1.0)
                    for task in reference
                ),
                "right_full_pass_left_not": sum(
                    math.isclose(native[right][task]["score"], 1.0)
                    and not math.isclose(native[left][task]["score"], 1.0)
                    for task in reference
                ),
            }

    tasks: list[dict[str, Any]] = []
    unique_top = Counter()
    co_top = Counter()
    for task in sorted(reference):
        scores = {harness: native[harness][task]["score"] for harness in HARNESSES}
        best = max(scores.values())
        top = [harness for harness, score in scores.items() if math.isclose(score, best)]
        if len(top) == 1:
            unique_top[top[0]] += 1
        for harness in top:
            co_top[harness] += 1
        tasks.append(
            {
                "task_id": task,
                "scenario_root": task_meta[task]["scenario_root"],
                "source": task_source(task_meta[task]["file"]),
                "complexity": task_meta[task]["complexity"],
                "scores": scores,
                "top_harnesses": top,
                "assistant_turns": {harness: traces[harness][task]["assistant_turns"] for harness in HARNESSES},
                "tool_calls": {harness: traces[harness][task]["tool_calls"] for harness in HARNESSES},
            }
        )

    scenario_scores: dict[str, Any] = {}
    scenarios: dict[str, list[str]] = defaultdict(list)
    for task in reference:
        scenarios[task_meta[task]["scenario_root"]].append(task)
    for scenario, task_ids in sorted(scenarios.items()):
        scenario_scores[scenario] = {
            "tasks": len(task_ids),
            "mean_scores": {
                harness: statistics.mean(native[harness][task]["score"] for task in task_ids)
                for harness in HARNESSES
            },
        }

    source_scores: dict[str, Any] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    for task in reference:
        sources[task_source(task_meta[task]["file"])].append(task)
    for source, task_ids in sorted(sources.items()):
        source_scores[source] = {
            "tasks": len(task_ids),
            "mean_scores": {
                harness: statistics.mean(native[harness][task]["score"] for task in task_ids)
                for harness in HARNESSES
            },
            "write_task_coverage": {
                harness: sum(traces[harness][task]["used_write"] for task in task_ids)
                for harness in HARNESSES
            },
        }

    hermes_runtime_failures = {
        "CTB_A02_investment_priority_matrix",
        "CTB_A03_cashflow_risk_memo",
    }
    runtime_clean_tasks = sorted(reference - hermes_runtime_failures)
    diagnostic_runtime_clean = {
        "excluded_tasks": sorted(hermes_runtime_failures),
        "reason": "Hermes terminal traceback reached the per-task process limit; post-hoc diagnostic only",
        "tasks": len(runtime_clean_tasks),
        "mean_scores": {
            harness: statistics.mean(native[harness][task]["score"] for task in runtime_clean_tasks)
            for harness in HARNESSES
        },
    }

    output = {
        "schema": "pawbench-trajectory-analysis/v1",
        "run_id": args.run_root.name,
        "task_count": len(reference),
        "trajectory_count": len(reference) * len(HARNESSES),
        "turn_definition": "one assistant-role message",
        "prompt_metadata": prompt_metadata(args.run_root),
        "harnesses": harness_summary,
        "pairwise": pairwise,
        "top_counts": {"unique_top": dict(unique_top), "including_ties": dict(co_top)},
        "scenario_scores": scenario_scores,
        "source_scores": source_scores,
        "diagnostic_runtime_clean": diagnostic_runtime_clean,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tasks": len(reference), "trajectories": len(reference) * 3}))


if __name__ == "__main__":
    main()
