#!/usr/bin/env python3
"""Clamp OpenAI-compatible generation requests for causal harness comparisons."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


GENERATION_KEYS = (
    "model",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "seed",
)


def normalize_chat_payload(
    payload: dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied payload with the causal-control fields overwritten."""
    normalized = dict(payload)
    original = {key: payload.get(key) for key in GENERATION_KEYS if key in payload}
    normalized["model"] = model
    normalized["temperature"] = temperature
    normalized["max_tokens"] = max_tokens
    normalized.pop("max_completion_tokens", None)
    return normalized, original


def append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def create_app(
    *,
    upstream_base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    audit_path: Path,
):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask

    app = FastAPI()
    client = httpx.AsyncClient(timeout=None)
    upstream = upstream_base_url.rstrip("/")

    @app.on_event("shutdown")
    async def close_client() -> None:
        await client.aclose()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request):
        request_id = uuid.uuid4().hex
        body = await request.body()
        original_generation: dict[str, Any] = {}
        effective_generation: dict[str, Any] = {}
        message_count = None
        tool_count = None
        stream = None

        if request.method == "POST" and path.rstrip("/") == "v1/chat/completions":
            payload = json.loads(body or b"{}")
            payload, original_generation = normalize_chat_payload(
                payload,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            body = json.dumps(payload, separators=(",", ":")).encode()
            effective_generation = {
                key: payload.get(key) for key in GENERATION_KEYS if key in payload
            }
            message_count = len(payload.get("messages") or [])
            tool_count = len(payload.get("tools") or [])
            stream = bool(payload.get("stream"))

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        upstream_request = client.build_request(
            request.method,
            f"{upstream}/{path}",
            headers=headers,
            content=body,
            params=request.query_params,
        )
        started = time.time()
        try:
            response = await client.send(upstream_request, stream=True)
        except Exception as exc:
            append_audit(
                audit_path,
                {
                    "request_id": request_id,
                    "time_unix": started,
                    "path": path,
                    "error": type(exc).__name__,
                },
            )
            raise

        append_audit(
            audit_path,
            {
                "request_id": request_id,
                "time_unix": started,
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "original_generation": original_generation,
                "effective_generation": effective_generation,
                "message_count": message_count,
                "tool_count": tool_count,
                "stream": stream,
            },
        )
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-base-url",
        default=os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:30000"),
    )
    parser.add_argument("--model", default=os.environ.get("CLAMP_MODEL", ""), required=False)
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("CLAMP_TEMPERATURE", "0")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("CLAMP_MAX_TOKENS", "8192")),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path(os.environ.get("CLAMP_AUDIT_PATH", "/audit/requests.jsonl")),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or CLAMP_MODEL is required")
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(
        upstream_base_url=args.upstream_base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        audit_path=args.audit_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
