#!/usr/bin/env python3
"""Create and verify a causal PawBench harness-gap pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "workplace-harness-pilot/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAWBENCH_PATCH = PROJECT_ROOT / "scripts/harness_adaptation/patches/pawbench-docker-network.patch"
PAWBENCH_PATCH_RELPATH = str(PAWBENCH_PATCH.relative_to(PROJECT_ROOT))
REQUEST_PROXY = PROJECT_ROOT / "scripts/harness_adaptation/openai_request_clamp_proxy.py"
REQUEST_PROXY_RELPATH = str(REQUEST_PROXY.relative_to(PROJECT_ROOT))
HARNESS_DOCKERFILES = {
    "qwenpaw": "docker/Dockerfile.pawbench-qwenpaw",
    "openclaw": "docker/Dockerfile.pawbench-openclaw",
    "hermes": "docker/Dockerfile.pawbench-hermes",
}
DEFAULT_IMAGES = {
    "qwenpaw": "qwenclawbench-qwenpaw:latest",
    "openclaw": "openclaw-pawbench:latest",
    "hermes": "hermes-qwenclawbench:latest",
}
RUNTIME_VERSION_COMMANDS = {
    "qwenpaw": ["qwenpaw", "--version"],
    "openclaw": ["openclaw", "--version"],
    "hermes": ["hermes", "--version"],
}
EXPECTED_RUNTIME_VERSIONS = {
    "qwenpaw": "1.1.3",
    # PawBench's 2026.4.24 npm dependency graph is currently unresolvable.
    # The compatibility build must remain explicit in the manifest.
    "openclaw": "2026.7.1",
    "hermes": "2026.4.23",
}
WORKPLACE_SCENARIOS = {
    "Office_Productivity",
    "Data_Analytics",
    "Content_Creation",
    "Information_Retrieval",
    "Knowledge",
    "Finance_Investment",
    "Legal",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"missing YAML frontmatter: {path}")
    try:
        raw = text.split("---\n", 2)[1]
    except IndexError as exc:
        raise ValueError(f"unterminated YAML frontmatter: {path}") from exc
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid YAML frontmatter: {path}")
    return parsed


def eligible_task(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8")
    meta = parse_frontmatter(path)
    labels = meta.get("labels") or {}
    modality = labels.get("modality") or {}
    scenario = str(labels.get("scenario") or "")
    scenario_root = scenario.split("/", 1)[0]
    if labels.get("environment") != "closed":
        return None
    if modality.get("type") != "text":
        return None
    if scenario_root not in WORKPLACE_SCENARIOS:
        return None
    if meta.get("grading_type") not in {"automated", "hybrid"}:
        return None
    if "## Automated Checks" not in text or "def grade(" not in text:
        return None
    return {
        "file": path.name,
        "task_id": str(meta["id"]),
        "name": str(meta.get("name") or meta["id"]),
        "scenario": scenario,
        "scenario_root": scenario_root,
        "complexity": labels.get("complexity"),
        "grading_type": meta["grading_type"],
        "timeout_seconds": int(meta.get("timeout_seconds", 300)),
        "sha256": sha256_file(path),
    }


def select_tasks(tasks_dir: Path, count: int) -> list[dict[str, Any]]:
    grouped: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for path in sorted(tasks_dir.glob("T*.md")):
        task = eligible_task(path)
        if task is not None:
            grouped[task["scenario_root"]].append(task)
    selected: list[dict[str, Any]] = []
    roots = sorted(grouped)
    while len(selected) < count:
        made_progress = False
        for root in roots:
            if grouped[root] and len(selected) < count:
                selected.append(grouped[root].popleft())
                made_progress = True
        if not made_progress:
            break
    if len(selected) != count:
        raise ValueError(f"need {count} eligible tasks, found {len(selected)}")
    return selected


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def docker_image_id(image: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def docker_runtime_version(image: str, harness: str) -> str | None:
    if docker_image_id(image) is None:
        return None
    result = subprocess.run(
        ["docker", "run", "--rm", image, *RUNTIME_VERSION_COMMANDS[harness]],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output or None


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.pawbench_root.resolve()
    tasks_dir = root / "data" / "pawbench-v1.0" / "tasks"
    if not tasks_dir.is_dir():
        raise ValueError(f"PawBench task directory missing: {tasks_dir}")
    if git_output(root, "status", "--porcelain"):
        raise ValueError("PawBench worktree must be clean before freezing a manifest")
    tasks = select_tasks(tasks_dir, args.task_count)
    harnesses = []
    for name in ("qwenpaw", "openclaw", "hermes"):
        dockerfile = root / HARNESS_DOCKERFILES[name]
        if not dockerfile.is_file():
            raise ValueError(f"missing harness Dockerfile: {dockerfile}")
        harnesses.append(
            {
                "name": name,
                "image": DEFAULT_IMAGES[name],
                "image_id": docker_image_id(DEFAULT_IMAGES[name]),
                "expected_runtime_version": EXPECTED_RUNTIME_VERSIONS[name],
                "observed_runtime_version": docker_runtime_version(DEFAULT_IMAGES[name], name),
                "dockerfile": HARNESS_DOCKERFILES[name],
                "dockerfile_sha256": sha256_file(dockerfile),
            }
        )
    task_identity = [{"file": t["file"], "sha256": t["sha256"]} for t in tasks]
    manifest = {
        "schema": SCHEMA,
        "state": "execution_ready"
        if all(h["image_id"] and h["observed_runtime_version"] for h in harnesses)
        else "images_pending",
        "benchmark": {
            "repo": "https://github.com/agentscope-ai/PawBench.git",
            "commit": git_output(root, "rev-parse", "HEAD"),
            "dataset": "pawbench-v1.0",
        },
        "model": {"id": args.model_id, "revision": args.model_revision},
        "shared_endpoint": {
            "base_url": args.base_url.rstrip("/"),
            "docker_network": args.docker_network,
            "context_limit": args.context_limit,
            "max_generated_tokens": args.max_generated_tokens,
            "temperature": args.temperature,
            "timeout_multiplier": args.timeout_multiplier,
        },
        "execution_overlay": {
            "patch": PAWBENCH_PATCH_RELPATH,
            "sha256": sha256_file(PAWBENCH_PATCH),
            "request_proxy": REQUEST_PROXY_RELPATH,
            "request_proxy_sha256": sha256_file(REQUEST_PROXY),
        },
        "primary_grading": "deterministic_automated_checks",
        "harnesses": harnesses,
        "tasks": tasks,
        "task_set_sha256": canonical_hash(task_identity),
        "pass_gate": {
            "paired_net_task_difference": 3,
            "trace_attributable_failures": 3,
        },
    }
    verify_manifest(manifest, root, require_images=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_manifest(manifest: dict[str, Any], root: Path, require_images: bool) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported manifest schema")
    names = [arm.get("name") for arm in manifest.get("harnesses", [])]
    if names != ["qwenpaw", "openclaw", "hermes"]:
        raise ValueError("manifest must contain exactly qwenpaw, openclaw, hermes in order")
    overlay = manifest.get("execution_overlay") or {}
    if overlay.get("patch") != PAWBENCH_PATCH_RELPATH:
        raise ValueError("unexpected PawBench execution overlay")
    if not PAWBENCH_PATCH.is_file() or sha256_file(PAWBENCH_PATCH) != overlay.get("sha256"):
        raise ValueError("PawBench execution overlay identity changed")
    if overlay.get("request_proxy") != REQUEST_PROXY_RELPATH:
        raise ValueError("unexpected request-clamp proxy")
    if not REQUEST_PROXY.is_file() or sha256_file(REQUEST_PROXY) != overlay.get("request_proxy_sha256"):
        raise ValueError("request-clamp proxy identity changed")
    endpoint = manifest.get("shared_endpoint") or {}
    if not endpoint.get("docker_network"):
        raise ValueError("shared Docker network is required")
    forbidden = {"model", "model_id", "base_url", "context_limit", "max_generated_tokens", "temperature"}
    for arm in manifest["harnesses"]:
        overlap = forbidden.intersection(arm)
        if overlap:
            raise ValueError(f"per-harness treatment override is forbidden: {arm['name']} {sorted(overlap)}")
        dockerfile = root / arm["dockerfile"]
        if sha256_file(dockerfile) != arm["dockerfile_sha256"]:
            raise ValueError(f"Dockerfile identity changed: {arm['name']}")
        if require_images:
            observed = docker_image_id(arm["image"])
            if not observed or observed != arm.get("image_id"):
                raise ValueError(f"container image identity missing or changed: {arm['name']}")
            version_output = docker_runtime_version(arm["image"], arm["name"])
            if not version_output or version_output != arm.get("observed_runtime_version"):
                raise ValueError(f"container runtime version missing or changed: {arm['name']}")
            if arm["expected_runtime_version"] not in version_output:
                raise ValueError(f"unexpected harness runtime version: {arm['name']}")
    tasks_dir = root / "data" / manifest["benchmark"]["dataset"] / "tasks"
    identity = []
    for task in manifest.get("tasks", []):
        path = tasks_dir / task["file"]
        if sha256_file(path) != task["sha256"]:
            raise ValueError(f"task identity changed: {task['file']}")
        identity.append({"file": task["file"], "sha256": task["sha256"]})
    if canonical_hash(identity) != manifest.get("task_set_sha256"):
        raise ValueError("task-set identity mismatch")
    if git_output(root, "rev-parse", "HEAD") != manifest["benchmark"]["commit"]:
        raise ValueError("PawBench commit changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate")
    create.add_argument("--pawbench-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--model-id", required=True)
    create.add_argument("--model-revision", required=True)
    create.add_argument("--base-url", required=True)
    create.add_argument("--docker-network", required=True)
    create.add_argument("--task-count", type=int, default=24)
    create.add_argument("--context-limit", type=int, default=32768)
    create.add_argument("--max-generated-tokens", type=int, default=8192)
    create.add_argument("--temperature", type=float, default=0.0)
    create.add_argument("--timeout-multiplier", type=float, default=1.0)
    check = sub.add_parser("verify")
    check.add_argument("--pawbench-root", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--require-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        manifest = generate(args)
        print(json.dumps({"state": manifest["state"], "tasks": len(manifest["tasks"]), "task_set_sha256": manifest["task_set_sha256"]}))
    else:
        manifest = json.loads(args.manifest.read_text())
        verify_manifest(manifest, args.pawbench_root.resolve(), args.require_images)
        print(json.dumps({"state": "verified", "require_images": args.require_images}))


if __name__ == "__main__":
    main()
