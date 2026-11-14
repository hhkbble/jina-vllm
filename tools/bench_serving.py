"""Measure warmed native HTTP latency/throughput on a frozen workload."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pynvml
from transformers import AutoTokenizer

from jina_vllm.reranker_v35.config import reranker_block_size
from jina_vllm.reranker_v35.ranking import prepare_request
from tools.compare_reference import model_directory, read_workload


async def wait_ready(client):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("Server readiness timed out")


async def served_max_model_len(client):
    response = await client.get("/v1/models")
    response.raise_for_status()
    return response.json()["data"][0]["max_model_len"]


async def measure(args):
    cases = read_workload(args.workload)
    workload = {
        "short-list": [c for c in cases if len(c["documents"]) <= 16][:3],
        "long-list": [c for c in cases if len(c["documents"]) >= 100][:3],
        "long-docs": sorted(
            [c for c in cases if len(c["documents"]) <= 64],
            key=lambda c: sum(map(len, c["documents"])) / len(c["documents"]),
            reverse=True,
        )[:3],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_directory(), local_files_only=True)
    block_size = reranker_block_size(args.additional_config)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        await wait_ready(client)
        max_model_len = await served_max_model_len(client)

    preprocessing = {}
    for name, selected in workload.items():
        samples = []
        for case in selected:
            start = time.perf_counter()
            prepare_request(
                [(case["query"], case["documents"])],
                tokenizer,
                max_model_len,
                block_size=block_size,
            )
            samples.append(time.perf_counter() - start)
        preprocessing[name] = {
            "p50_seconds": float(np.median(samples)),
            "samples": samples,
            "scope": "isolated planner; not server CPU profiling",
        }
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    cells = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        await wait_ready(client)

        async def request(case):
            start = time.perf_counter()
            response = await client.post(
                "/rerank",
                json={
                    "query": case["query"],
                    "documents": case["documents"],
                },
            )
            response.raise_for_status()
            result = response.json()
            return dict(
                id=case["id"],
                seconds=time.perf_counter() - start,
                documents=len(case["documents"]),
                tokens=result["usage"]["prompt_tokens"],
            )

        for name, selected in workload.items():
            for case in selected:
                await request(case)
            for concurrency in args.concurrency:
                semaphore = asyncio.Semaphore(concurrency)
                peak = pynvml.nvmlDeviceGetMemoryInfo(gpu).used

                async def limited(case, semaphore=semaphore):
                    async with semaphore:
                        return await request(case)

                await asyncio.gather(
                    *(
                        limited(selected[i % len(selected)])
                        for i in range(args.warmup_requests)
                    )
                )
                stop = asyncio.Event()
                gpu_samples = []
                start = time.perf_counter()

                async def monitor(stop=stop, gpu_samples=gpu_samples, start=start):
                    while not stop.is_set():
                        gpu_samples.append(
                            dict(
                                seconds=time.perf_counter() - start,
                                memory_used_bytes=pynvml.nvmlDeviceGetMemoryInfo(
                                    gpu
                                ).used,
                                gpu_util_percent=pynvml.nvmlDeviceGetUtilizationRates(
                                    gpu
                                ).gpu,
                            )
                        )
                        await asyncio.sleep(0.02)

                monitoring = asyncio.create_task(monitor())
                try:
                    samples = await asyncio.gather(
                        *(
                            limited(selected[i % len(selected)])
                            for i in range(args.requests)
                        )
                    )
                    elapsed = time.perf_counter() - start
                finally:
                    stop.set()
                    await monitoring
                latency = np.array([s["seconds"] for s in samples])
                cell = dict(
                    workload=name,
                    concurrency=concurrency,
                    count=len(samples),
                    avg_seconds=float(latency.mean()),
                    p50_seconds=float(np.median(latency)),
                    p95_seconds=float(np.percentile(latency, 95)),
                    requests_per_second=len(samples) / elapsed,
                    documents_per_second=sum(s["documents"] for s in samples) / elapsed,
                    prompt_tokens_per_second=sum(s["tokens"] for s in samples)
                    / elapsed,
                    gpu_peak_device_bytes=max(
                        [peak, *(s["memory_used_bytes"] for s in gpu_samples)]
                    ),
                    gpu_util_avg_percent=float(
                        np.mean([s["gpu_util_percent"] for s in gpu_samples])
                    ),
                    gpu_samples=gpu_samples,
                    samples=samples,
                )
                cells.append(cell)
                print(
                    f"{name} c={concurrency}: avg {cell['avg_seconds']:.4f}s "
                    f"p50 {cell['p50_seconds']:.4f}s "
                    f"p95 {cell['p95_seconds']:.4f}s",
                    flush=True,
                )
    pynvml.nvmlShutdown()
    return dict(
        mode=args.mode,
        block_size=block_size,
        max_model_len=max_model_len,
        workload=str(args.workload),
        cells=cells,
        preprocessing=preprocessing,
        warmup_requests_per_cell=args.warmup_requests,
        notes=(
            "Warmup per input plus optional concurrent warmup. NVML device "
            "samples every 20 ms during timing, including other device users. "
            "Latency excludes client semaphore waiting and includes server queueing."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bf16", "fp8"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18035")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-config", type=json.loads, default={})
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--warmup-requests", type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(measure(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
