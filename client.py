#!/usr/bin/env python3

"""Configurable-workload throughput benchmark for a Kimi K3 replica pool."""

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from runtime import http_request, load_workers, make_opener, require_http_200


@dataclass(frozen=True)
class RequestResult:
    worker_index: int
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class CaseResult:
    server_count: int
    concurrency_per_server: int
    request_count: int
    elapsed_seconds: float
    output_tokens_per_second: float
    input_tokens_per_second: float
    total_tokens_per_second: float
    request_throughput: float
    mean_latency_seconds: float
    median_latency_seconds: float
    p95_latency_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--concurrency-per-server", type=int, required=True)
    parser.add_argument("--waves", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_vocab_size(model_dir: Path) -> int:
    config = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("model config has no text_config")
    vocab_size = text_config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 2000:
        raise RuntimeError(f"invalid vocab_size: {vocab_size}")
    return vocab_size


def input_ids(
    token_count: int,
    vocab_size: int,
    seed: int,
    request_index: int,
) -> list[int]:
    generator = random.Random(seed + request_index * 1_000_003)
    return [
        generator.randrange(1000, min(vocab_size, 50000))
        for _ in range(token_count)
    ]


def generate(
    worker_index: int,
    worker_url: str,
    tokens: list[int],
    output_tokens: int,
    timeout_seconds: float,
) -> RequestResult:
    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
    }
    start = time.monotonic()
    result = http_request(
        make_opener(),
        "POST",
        f"{worker_url}/generate",
        timeout_seconds,
        payload,
    )
    response = require_http_200(result, f"worker {worker_index} /generate")
    elapsed = time.monotonic() - start
    if not isinstance(response, dict) or not isinstance(
        response.get("meta_info"), dict
    ):
        raise RuntimeError(f"worker {worker_index} returned no meta_info")
    meta = response["meta_info"]
    prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")
    if prompt_tokens != len(tokens) or completion_tokens != output_tokens:
        raise RuntimeError(
            f"worker {worker_index} token mismatch: "
            f"prompt={prompt_tokens}/{len(tokens)} "
            f"completion={completion_tokens}/{output_tokens}"
        )
    return RequestResult(
        worker_index,
        elapsed,
        prompt_tokens,
        completion_tokens,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def run_case(
    worker_urls: list[str],
    server_count: int,
    concurrency_per_server: int,
    waves: int,
    input_token_count: int,
    output_token_count: int,
    vocab_size: int,
    seed: int,
    timeout_seconds: float,
) -> CaseResult:
    concurrency = server_count * concurrency_per_server
    request_count = concurrency * waves
    requests: list[tuple[int, str, list[int]]] = []

    request_index = 0
    for _wave in range(waves):
        for _slot in range(concurrency_per_server):
            for worker_index in range(server_count):
                requests.append(
                    (
                        worker_index,
                        worker_urls[worker_index],
                        input_ids(
                            input_token_count,
                            vocab_size,
                            seed,
                            request_index,
                        ),
                    )
                )
                request_index += 1

    start = time.monotonic()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                generate,
                worker_index,
                worker_url,
                tokens,
                output_token_count,
                timeout_seconds,
            )
            for worker_index, worker_url, tokens in requests
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed = time.monotonic() - start

    if len(results) != request_count:
        raise RuntimeError(
            f"expected {request_count} results; got {len(results)}"
        )
    expected_per_worker = concurrency_per_server * waves
    for worker_index in range(server_count):
        actual = sum(
            result.worker_index == worker_index for result in results
        )
        if actual != expected_per_worker:
            raise RuntimeError(
                f"worker {worker_index} received {actual} requests; "
                f"expected {expected_per_worker}"
            )

    total_input = sum(result.prompt_tokens for result in results)
    total_output = sum(result.completion_tokens for result in results)
    latencies = [result.latency_seconds for result in results]
    return CaseResult(
        server_count=server_count,
        concurrency_per_server=concurrency_per_server,
        request_count=request_count,
        elapsed_seconds=elapsed,
        output_tokens_per_second=total_output / elapsed,
        input_tokens_per_second=total_input / elapsed,
        total_tokens_per_second=(total_input + total_output) / elapsed,
        request_throughput=request_count / elapsed,
        mean_latency_seconds=statistics.fmean(latencies),
        median_latency_seconds=statistics.median(latencies),
        p95_latency_seconds=percentile(latencies, 0.95),
    )


def main() -> None:
    args = parse_args()
    positive_values = (
        args.expected_workers,
        args.concurrency_per_server,
        args.waves,
        args.input_tokens,
        args.output_tokens,
        args.timeout_seconds,
    )
    if any(value <= 0 for value in positive_values):
        raise SystemExit(
            "ERROR: worker, concurrency, wave, token, and timeout values "
            "must be positive"
        )
    if args.seed < 0:
        raise SystemExit("ERROR: seed must be non-negative")

    router_url = args.router_url.rstrip("/")
    worker_urls = load_workers(
        make_opener(),
        router_url,
        args.expected_workers,
        args.timeout_seconds,
    )
    vocab_size = load_vocab_size(args.model_dir)
    server_counts = list(range(1, args.expected_workers + 1))
    seed_generator = random.Random(args.seed)

    print(f"router_url={router_url}")
    print(f"worker_count={len(worker_urls)}")
    print(f"server_counts={','.join(map(str, server_counts))}")
    print(f"concurrency_per_server={args.concurrency_per_server}")
    print(f"waves={args.waves}")
    print(f"input_tokens={args.input_tokens}")
    print(f"output_tokens={args.output_tokens}")
    print(f"seed={args.seed}")
    print(f"result_json={args.output_json}")
    for index, worker_url in enumerate(worker_urls):
        print(f"worker index={index} url={worker_url}")

    print("=== Initial warmup ===", flush=True)
    run_case(
        worker_urls,
        args.expected_workers,
        1,
        1,
        args.input_tokens,
        args.output_tokens,
        vocab_size,
        seed_generator.randrange(1, 2**63),
        args.timeout_seconds,
    )

    cases: list[CaseResult] = []
    for server_count in server_counts:
        print(f"case_warmup servers={server_count}", flush=True)
        run_case(
            worker_urls,
            server_count,
            args.concurrency_per_server,
            1,
            args.input_tokens,
            args.output_tokens,
            vocab_size,
            seed_generator.randrange(1, 2**63),
            args.timeout_seconds,
        )
        print(f"case_start servers={server_count}", flush=True)
        result = run_case(
            worker_urls,
            server_count,
            args.concurrency_per_server,
            args.waves,
            args.input_tokens,
            args.output_tokens,
            vocab_size,
            seed_generator.randrange(1, 2**63),
            args.timeout_seconds,
        )
        cases.append(result)
        print(
            f"case_result servers={server_count} "
            f"requests={result.request_count} "
            f"elapsed_seconds={result.elapsed_seconds:.3f} "
            f"output_tokens_per_second={result.output_tokens_per_second:.3f} "
            f"input_tokens_per_second={result.input_tokens_per_second:.3f} "
            f"total_tokens_per_second={result.total_tokens_per_second:.3f} "
            f"request_throughput={result.request_throughput:.3f} "
            f"mean_latency_seconds={result.mean_latency_seconds:.3f} "
            f"p95_latency_seconds={result.p95_latency_seconds:.3f}",
            flush=True,
        )

    baseline = cases[0].output_tokens_per_second
    output = {
        "router_url": router_url,
        "worker_urls": worker_urls,
        "input_tokens_per_request": args.input_tokens,
        "output_tokens_per_request": args.output_tokens,
        "concurrency_per_server": args.concurrency_per_server,
        "waves": args.waves,
        "seed": args.seed,
        "cases": [],
    }
    for case in cases:
        case_output = asdict(case)
        case_output["speedup"] = case.output_tokens_per_second / baseline
        case_output["scaling_efficiency"] = (
            case.output_tokens_per_second
            / (baseline * case.server_count)
        )
        output["cases"].append(case_output)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"K3_SERVER_SCALING_OK cases={len(cases)} "
        f"result_json={args.output_json}"
    )


if __name__ == "__main__":
    main()
