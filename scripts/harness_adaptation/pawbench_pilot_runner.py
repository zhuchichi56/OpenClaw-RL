#!/usr/bin/env python3
"""Execute frozen PawBench pilot arms with one shared model endpoint and budget."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pawbench_pilot_manifest import PROJECT_ROOT, verify_manifest


PATCHED_FILES = {
    "docker/Dockerfile.pawbench-qwenpaw",
    "pawbench/agents/impl/hermes_agent.py",
    "pawbench/agents/impl/openclaw_agent.py",
    "pawbench/agents/impl/qwenpaw_agent.py",
    "pawbench/agents/factory.py",
    "pawbench/backend.py",
    "pawbench/envs/docker.py",
    "pawbench/grader.py",
}


def command_output(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).rstrip("\n")


def ensure_execution_overlay(manifest: dict[str, Any], pawbench_root: Path) -> None:
    patch_path = PROJECT_ROOT / manifest["execution_overlay"]["patch"]
    status = command_output("git", "status", "--porcelain", cwd=pawbench_root)
    if not status:
        subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=pawbench_root,
            check=True,
        )
        subprocess.run(["git", "apply", str(patch_path)], cwd=pawbench_root, check=True)
        status = command_output("git", "status", "--porcelain", cwd=pawbench_root)

    changed = {
        line[3:] for line in status.splitlines() if len(line) >= 4 and not line.startswith("??")
    }
    untracked = [line for line in status.splitlines() if line.startswith("??")]
    if changed != PATCHED_FILES or untracked:
        raise RuntimeError(
            f"PawBench checkout has changes outside the frozen overlay: {status}"
        )
    subprocess.run(
        ["git", "apply", "--check", "-R", str(patch_path)],
        cwd=pawbench_root,
        check=True,
    )


def verify_checkout_and_overlay(
    manifest: dict[str, Any], pawbench_root: Path
) -> None:
    patch_path = PROJECT_ROOT / manifest["execution_overlay"]["patch"]
    status = command_output("git", "status", "--porcelain", cwd=pawbench_root)
    if status:
        ensure_execution_overlay(manifest, pawbench_root)
        subprocess.run(["git", "apply", "-R", str(patch_path)], cwd=pawbench_root, check=True)
        try:
            verify_manifest(manifest, pawbench_root, require_images=True)
        finally:
            subprocess.run(["git", "apply", str(patch_path)], cwd=pawbench_root, check=True)
    else:
        verify_manifest(manifest, pawbench_root, require_images=True)
    ensure_execution_overlay(manifest, pawbench_root)


def endpoint_preflight(manifest: dict[str, Any]) -> None:
    endpoint = manifest["shared_endpoint"]
    qwenpaw_image = next(
        arm["image"] for arm in manifest["harnesses"] if arm["name"] == "qwenpaw"
    )
    probe = (
        "import json,urllib.request; "
        f"d=json.load(urllib.request.urlopen({(endpoint['base_url'] + '/models')!r}, timeout=15)); "
        "assert d.get('data'), d; print(d['data'][0]['id'])"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            endpoint["docker_network"],
            qwenpaw_image,
            "python",
            "-c",
            probe,
        ],
        check=True,
    )


def build_agent_config(
    manifest: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any]:
    endpoint = manifest["shared_endpoint"]
    return {
        "model": manifest["model"]["id"],
        "api_key": "pawbench-local-pilot",
        "base_url": endpoint["base_url"],
        "timeout_multiplier": endpoint["timeout_multiplier"],
        "docker_image": arm["image"],
        "docker_network": endpoint["docker_network"],
        "agent_type": arm["name"],
        "context_limit": endpoint["context_limit"],
        "max_tokens": endpoint["max_generated_tokens"],
        "generate_kwargs": {
            "temperature": endpoint["temperature"],
            "max_tokens": endpoint["max_generated_tokens"],
        },
        "save_workspace": True,
        "automated_only_grading": True,
        "verbose": True,
    }


async def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pawbench_root = args.pawbench_root.resolve()
    verify_checkout_and_overlay(manifest, pawbench_root)
    endpoint_preflight(manifest)

    if args.all_manifest_tasks:
        selected_tasks = list(manifest["tasks"])
    else:
        selected = next(
            (task for task in manifest["tasks"] if task["file"] == args.task), None
        )
        if selected is None:
            raise ValueError(f"task is not in the frozen manifest: {args.task}")
        selected_tasks = [selected]
    task_filter = [Path(task["file"]).stem for task in selected_tasks]

    sys.path.insert(0, str(pawbench_root))
    from pawbench import BenchmarkRunner, PawBenchBackend

    by_name = {arm["name"]: arm for arm in manifest["harnesses"]}
    summaries: list[dict[str, Any]] = []
    for agent_name in args.agents:
        if agent_name not in by_name:
            raise ValueError(f"agent is not in the frozen manifest: {agent_name}")
        arm_dir = args.output.resolve() / agent_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        config = build_agent_config(manifest, by_name[agent_name])
        (arm_dir / "execution_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        runner = BenchmarkRunner(
            backend=PawBenchBackend(pawbench_root),
            results_dir=arm_dir,
            concurrency=1,
            max_retries=1,
            runs_per_task=1,
        )
        results = await runner.run(
            agent_config=config,
            task_filter=task_filter,
            dataset=manifest["benchmark"]["dataset"],
        )
        if len(results) != len(selected_tasks):
            raise RuntimeError(
                f"expected {len(selected_tasks)} results for {agent_name}, got {len(results)}"
            )
        for native_result in results:
            result = dataclasses.asdict(native_result)
            summary = {
                key: result[key]
                for key in (
                    "task_id",
                    "score",
                    "max_score",
                    "passed",
                    "status",
                    "execution_time",
                    "transcript_length",
                    "anomaly",
                )
            }
            summary["agent"] = agent_name
            summaries.append(summary)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pawbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task")
    selection.add_argument("--all-manifest-tasks", action="store_true")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["qwenpaw", "openclaw", "hermes"],
    )
    return parser.parse_args()


def main() -> None:
    summaries = asyncio.run(execute(parse_args()))
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
