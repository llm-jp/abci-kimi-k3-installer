#!/usr/bin/env python3

"""Runtime checks and readiness helpers for the ABCI Kimi K3 deployment."""

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUST_EXTENSION_MODULES = (
    "sglang.srt.grpc._core",
    "sglang.srt.multimodal._core",
    "sglang.srt.server._core",
)


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


@dataclass(frozen=True)
class Replica:
    index: int
    leader_host: str
    endpoint: str
    launcher_pid: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_model = subparsers.add_parser("verify-model")
    verify_model.add_argument("--model-dir", type=Path, required=True)

    verify_install = subparsers.add_parser("verify-install")
    verify_install.add_argument("--source-dir", type=Path, required=True)
    verify_install.add_argument("--expected-commit", required=True)
    verify_install.add_argument("--router-version", required=True)

    prewarm = subparsers.add_parser("prewarm")
    prewarm.add_argument("--model-dir", type=Path, required=True)

    wait_replicas = subparsers.add_parser("wait-replicas")
    wait_replicas.add_argument("--manifest", type=Path, required=True)
    wait_replicas.add_argument("--timeout-seconds", type=int, required=True)

    wait_router = subparsers.add_parser("wait-router")
    wait_router.add_argument("--endpoint", required=True)
    wait_router.add_argument("--pid", type=int, required=True)
    wait_router.add_argument("--expected-workers", type=int, required=True)
    wait_router.add_argument("--timeout-seconds", type=int, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--base-url", required=True)
    smoke.add_argument("--timeout-seconds", type=float, default=600.0)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--router-url", required=True)
    capacity.add_argument("--model-dir", type=Path, required=True)
    capacity.add_argument("--expected-workers", type=int, required=True)
    capacity.add_argument("--timeout-seconds", type=float, default=60.0)

    return parser


def make_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_request(
    opener: urllib.request.OpenerDirector,
    method: str,
    url: str,
    timeout_seconds: float,
    payload: dict[str, Any] | None = None,
) -> HttpResult:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return HttpResult(response.status, response.read())
    except urllib.error.HTTPError as error:
        return HttpResult(error.code, error.read())
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"{method} {url} failed: {error}") from error


def decode_json(result: HttpResult, label: str) -> Any:
    text = result.body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} returned invalid JSON: {text[:2000]}") from error


def one_line(value: Any, limit: int = 1600) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = "" if value is None else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def require_http_200(result: HttpResult, label: str) -> Any:
    if result.status != 200:
        raise RuntimeError(
            f"{label} returned HTTP {result.status}: {one_line(result.body)}"
        )
    return decode_json(result, label)


def load_workers(
    opener: urllib.request.OpenerDirector,
    router_url: str,
    expected_workers: int,
    timeout_seconds: float,
) -> list[str]:
    result = http_request(
        opener,
        "GET",
        f"{router_url.rstrip('/')}/workers",
        timeout_seconds,
    )
    response = require_http_200(result, "Router /workers")
    if not isinstance(response, dict) or not isinstance(
        response.get("workers"), list
    ):
        raise RuntimeError("Router /workers did not return a worker list")

    workers = response["workers"]
    if len(workers) != expected_workers:
        raise RuntimeError(
            f"expected {expected_workers} workers; found {len(workers)}"
        )

    worker_urls: list[str] = []
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict):
            raise RuntimeError(f"worker {index} is not an object")
        worker_url = worker.get("url")
        health = worker.get("is_healthy", worker.get("healthy"))
        if not isinstance(worker_url, str) or not worker_url:
            raise RuntimeError(f"worker {index} has no URL")
        if health is not True:
            raise RuntimeError(f"worker {index} is not healthy")
        worker_urls.append(worker_url.rstrip("/"))

    if len(set(worker_urls)) != len(worker_urls):
        raise RuntimeError(f"duplicate worker URLs: {worker_urls}")
    return worker_urls


def command_verify_model(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise SystemExit(f"ERROR: required model file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "kimi_k3":
        raise SystemExit(
            f"ERROR: unexpected model_type: {config.get('model_type')}"
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("ERROR: model index has no weight_map")

    shard_names = sorted(set(weight_map.values()))
    missing = []
    broken_symlinks = []
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise SystemExit("ERROR: model index contains a non-string shard")
        shard_path = model_dir / shard_name
        if shard_path.is_symlink() and not shard_path.exists():
            broken_symlinks.append(str(shard_path))
        elif not shard_path.is_file():
            missing.append(str(shard_path))

    print(f"model_dir={model_dir}")
    print("model_type=kimi_k3")
    print(f"referenced_shards={len(shard_names)}")
    print(f"missing_shards={len(missing)}")
    print(f"broken_symlinks={len(broken_symlinks)}")
    if missing or broken_symlinks:
        for path in missing:
            print(f"MISSING {path}")
        for path in broken_symlinks:
            print(f"BROKEN_SYMLINK {path}")
        raise SystemExit("ERROR: model weights are incomplete")
    print("K3_MODEL_OK")


def is_extension_module(path: Path) -> bool:
    return any(
        str(path).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def command_verify_install(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != args.expected_commit:
        raise SystemExit(
            f"ERROR: source commit mismatch: {actual_commit}"
        )

    expected_package_root = (source_dir / "python" / "sglang").resolve()
    sglang = importlib.import_module("sglang")
    if not sglang.__file__:
        raise SystemExit("ERROR: sglang has no module path")
    sglang_path = Path(sglang.__file__).resolve()
    if expected_package_root not in sglang_path.parents:
        raise SystemExit(
            f"ERROR: sglang was imported from {sglang_path}; "
            f"expected {expected_package_root}"
        )

    extension_paths: list[Path] = []
    for module_name in RUST_EXTENSION_MODULES:
        module = importlib.import_module(module_name)
        origin = getattr(module.__spec__, "origin", None)
        if not origin:
            raise SystemExit(f"ERROR: missing module origin: {module_name}")
        path = Path(origin).resolve()
        if not is_extension_module(path):
            raise SystemExit(f"ERROR: not a native extension: {path}")
        extension_paths.append(path)

    torch = importlib.import_module("torch")
    if torch.__version__ != "2.11.0+cu130":
        raise SystemExit(
            f"ERROR: torch version mismatch: {torch.__version__}"
        )
    if torch.version.cuda != "13.0":
        raise SystemExit(
            f"ERROR: torch CUDA version mismatch: {torch.version.cuda}"
        )
    importlib.import_module("flashinfer")
    importlib.import_module("sglang.srt.configs.kimi_k3")
    router = importlib.import_module("sglang_router")
    router_extension = importlib.import_module(
        "sglang_router.sglang_router_rs"
    )

    required_versions = {
        "flashinfer-python": "0.6.15.post1",
        "sgl-deep-gemm": "0.1.5",
        "sglang-router": args.router_version,
    }
    for distribution_name, expected_version in required_versions.items():
        actual_version = importlib.metadata.version(distribution_name)
        if actual_version != expected_version:
            raise SystemExit(
                f"ERROR: {distribution_name} version mismatch: "
                f"{actual_version} != {expected_version}"
            )

    router_extension_path = Path(router_extension.__file__).resolve()
    if not is_extension_module(router_extension_path):
        raise SystemExit(
            f"ERROR: Router Rust extension is invalid: {router_extension_path}"
        )

    print(f"source_commit={actual_commit}")
    print(f"sglang={importlib.metadata.version('sglang')}")
    print(f"sglang_file={sglang_path}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"rust_extension_count={len(extension_paths)}")
    print(f"router={router.__file__}")
    print(f"router_extension={router_extension_path}")
    for name, version in required_versions.items():
        print(f"{name}={version}")
    print("K3_INSTALL_VERIFY_OK")


def command_prewarm(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"ERROR: model config not found: {model_dir}")
    modules_cache = os.environ.get("HF_MODULES_CACHE")
    if not modules_cache:
        raise SystemExit("ERROR: HF_MODULES_CACHE is not set")

    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    print(f"transformers={transformers.__version__}")
    print(f"hf_modules_cache={modules_cache}")
    print(
        f"processor={processor.__class__.__module__}."
        f"{processor.__class__.__name__}"
    )
    if tokenizer is not None:
        print(
            f"tokenizer={tokenizer.__class__.__module__}."
            f"{tokenizer.__class__.__name__}"
        )
    print("K3_TRANSFORMERS_CACHE_PREWARM_OK")


def process_state(process_id: int) -> str | None:
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        return "unknown"
    fields = stat_text[closing_parenthesis + 1 :].split()
    return fields[0] if fields else "unknown"


def load_replica_manifest(path: Path) -> list[Replica]:
    replicas: list[Replica] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise SystemExit(
                f"ERROR: invalid manifest line {line_number}: {line}"
            )
        index, leader_host, endpoint, pid = fields
        replicas.append(
            Replica(
                index=int(index),
                leader_host=leader_host,
                endpoint=endpoint,
                launcher_pid=int(pid),
            )
        )
    replicas.sort(key=lambda replica: replica.index)
    if [replica.index for replica in replicas] != list(range(len(replicas))):
        raise SystemExit("ERROR: replica manifest indices are not contiguous")
    if not replicas:
        raise SystemExit("ERROR: replica manifest is empty")
    return replicas


def command_wait_replicas(args: argparse.Namespace) -> None:
    replicas = load_replica_manifest(args.manifest)
    opener = make_opener()
    ready: set[int] = set()
    last_errors = {replica.index: "not attempted" for replica in replicas}
    start = time.monotonic()
    deadline = start + args.timeout_seconds
    next_report = start

    print(f"replica_count={len(replicas)}", flush=True)
    print(f"ready_timeout_seconds={args.timeout_seconds}", flush=True)
    while time.monotonic() < deadline:
        for replica in replicas:
            if replica.index in ready:
                continue
            state = process_state(replica.launcher_pid)
            if state is None or state == "Z":
                raise SystemExit(
                    f"ERROR: replica {replica.index} launcher exited "
                    "before readiness"
                )
            try:
                result = http_request(
                    opener,
                    "GET",
                    replica.endpoint,
                    5,
                )
                if result.status == 200:
                    ready.add(replica.index)
                    print(
                        f"replica_ready index={replica.index} "
                        f"leader={replica.leader_host} "
                        f"elapsed_seconds={time.monotonic() - start:.1f}",
                        flush=True,
                    )
                else:
                    last_errors[replica.index] = f"HTTP {result.status}"
            except RuntimeError as error:
                last_errors[replica.index] = str(error)

        if len(ready) == len(replicas):
            print(
                f"all_ready_elapsed_seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
            print(f"K3_REPLICAS_READY count={len(replicas)}", flush=True)
            return

        now = time.monotonic()
        if now >= next_report:
            pending = [
                f"{replica.index}:{process_state(replica.launcher_pid)}:"
                f"{one_line(last_errors[replica.index], 120)}"
                for replica in replicas
                if replica.index not in ready
            ]
            print(
                f"waiting elapsed_seconds={now - start:.0f} "
                f"ready={len(ready)}/{len(replicas)} "
                f"pending={' | '.join(pending)}",
                flush=True,
            )
            next_report = now + 60
        time.sleep(10)

    pending = sorted(set(range(len(replicas))) - ready)
    raise SystemExit(
        f"ERROR: replica readiness timed out; pending={pending}"
    )


def command_wait_router(args: argparse.Namespace) -> None:
    opener = make_opener()
    start = time.monotonic()
    deadline = start + args.timeout_seconds
    next_report = start
    last_error = "not attempted"

    while time.monotonic() < deadline:
        state = process_state(args.pid)
        if state is None or state == "Z":
            raise SystemExit("ERROR: Router exited before readiness")
        try:
            result = http_request(opener, "GET", args.endpoint, 5)
            if result.status == 200:
                response = decode_json(result, "Router readiness")
                if (
                    isinstance(response, dict)
                    and response.get("healthy_workers")
                    == args.expected_workers
                    and response.get("total_workers")
                    == args.expected_workers
                ):
                    print(
                        f"router_ready_body={one_line(result.body)}",
                        flush=True,
                    )
                    print(
                        f"router_ready_elapsed_seconds="
                        f"{time.monotonic() - start:.1f}",
                        flush=True,
                    )
                    print("K3_ROUTER_READY", flush=True)
                    return
                last_error = f"not all workers ready: {one_line(result.body)}"
            else:
                last_error = f"HTTP {result.status}: {one_line(result.body)}"
        except RuntimeError as error:
            last_error = str(error)

        now = time.monotonic()
        if now >= next_report:
            print(
                f"router_waiting elapsed_seconds={now - start:.0f} "
                f"state={state} last_error={one_line(last_error, 180)}",
                flush=True,
            )
            next_report = now + 15
        time.sleep(2)

    raise SystemExit(
        f"ERROR: Router readiness timed out: {one_line(last_error)}"
    )


def command_smoke(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        raise SystemExit("ERROR: --base-url must end in /v1")
    opener = make_opener()

    models_result = http_request(
        opener,
        "GET",
        f"{base_url}/models",
        args.timeout_seconds,
    )
    models_response = require_http_200(models_result, "/v1/models")
    models = models_response.get("data") if isinstance(models_response, dict) else None
    if not isinstance(models, list) or not models:
        raise SystemExit("ERROR: /models returned no model")
    model_id = models[0].get("id")
    if not isinstance(model_id, str):
        raise SystemExit("ERROR: /models returned an invalid model")

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ],
        "reasoning_effort": "low",
        "max_tokens": 32,
        "stream": False,
    }
    start = time.monotonic()
    chat_result = http_request(
        opener,
        "POST",
        f"{base_url}/chat/completions",
        args.timeout_seconds,
        payload,
    )
    chat = require_http_200(chat_result, "/v1/chat/completions")
    choices = chat.get("choices") if isinstance(chat, dict) else None
    if not isinstance(choices, list) or not choices:
        raise SystemExit("ERROR: chat completion returned no choices")
    message = choices[0].get("message")
    usage = chat.get("usage", {})
    if not isinstance(message, dict) or not message.get("content"):
        raise SystemExit("ERROR: chat completion returned no content")

    print(f"model_id={model_id}")
    print(f"latency_seconds={time.monotonic() - start:.2f}")
    print(f"prompt_tokens={usage.get('prompt_tokens')}")
    print(f"completion_tokens={usage.get('completion_tokens')}")
    print(f"content={one_line(message.get('content'))}")
    print("K3_ROUTER_API_OK")


def command_capacity(args: argparse.Namespace) -> None:
    if args.expected_workers <= 0:
        raise SystemExit("ERROR: --expected-workers must be positive")
    opener = make_opener()
    router_url = args.router_url.rstrip("/")
    worker_urls = load_workers(
        opener,
        router_url,
        args.expected_workers,
        args.timeout_seconds,
    )
    target_worker = worker_urls[0]

    info_result = http_request(
        opener,
        "GET",
        f"{target_worker}/server_info",
        args.timeout_seconds,
    )
    info = require_http_200(info_result, "worker /server_info")
    if not isinstance(info, dict):
        raise SystemExit("ERROR: /server_info returned a non-object")
    max_input = info.get("max_req_input_len")
    max_total = info.get("max_total_num_tokens")
    auto_truncate = info.get("allow_auto_truncate")
    if (
        not isinstance(max_input, int)
        or not isinstance(max_total, int)
        or max_input <= 0
        or max_input >= max_total
    ):
        raise SystemExit("ERROR: invalid runtime token limits")
    if auto_truncate is not False:
        raise SystemExit(
            "ERROR: capacity check requires allow_auto_truncate=false"
        )

    config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    model_context = config.get("text_config", {}).get(
        "max_position_embeddings"
    )

    sampling = {
        "temperature": 0.0,
        "max_new_tokens": 1,
        "ignore_eos": True,
    }
    control = {
        "input_ids": [1000] * 32,
        "sampling_params": sampling,
    }
    control_result = http_request(
        opener,
        "POST",
        f"{target_worker}/generate",
        args.timeout_seconds,
        control,
    )
    if control_result.status != 200:
        raise SystemExit(
            f"ERROR: capacity control failed: HTTP {control_result.status}"
        )

    rejected_tokens = max_input + 1
    rejected_payload = {
        "input_ids": [1000] * rejected_tokens,
        "sampling_params": sampling,
    }
    rejected_result = http_request(
        opener,
        "POST",
        f"{target_worker}/generate",
        args.timeout_seconds,
        rejected_payload,
    )
    if not 400 <= rejected_result.status < 500:
        raise SystemExit(
            "ERROR: over-limit request was not rejected with HTTP 4xx"
        )

    health_result = http_request(
        opener,
        "GET",
        f"{target_worker}/health",
        args.timeout_seconds,
    )
    readiness_result = http_request(
        opener,
        "GET",
        f"{router_url}/readiness",
        args.timeout_seconds,
    )
    readiness = require_http_200(readiness_result, "Router readiness")
    if (
        health_result.status != 200
        or not isinstance(readiness, dict)
        or readiness.get("healthy_workers") != args.expected_workers
    ):
        raise SystemExit("ERROR: service health changed after capacity check")

    print(f"target_worker={target_worker}")
    print(f"model_context_tokens={model_context}")
    print(f"max_total_num_tokens={max_total}")
    print(f"max_req_input_len={max_input}")
    print(f"reserved_token_slots={max_total - max_input}")
    print(f"rejected_input_tokens={rejected_tokens}")
    print(f"rejected_http_status={rejected_result.status}")
    print(f"rejected_body={one_line(rejected_result.body)}")
    print("K3_TOKEN_CAPACITY_BOUNDARY_OK")


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "verify-model": command_verify_model,
        "verify-install": command_verify_install,
        "prewarm": command_prewarm,
        "wait-replicas": command_wait_replicas,
        "wait-router": command_wait_router,
        "smoke": command_smoke,
        "capacity": command_capacity,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
