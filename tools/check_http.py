"""Exercise real native routes and socket disconnect cleanup."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import torch

from tools.bench_serving import wait_ready
from tools.compare_reference import read_workload


def metric(text, name, reason=None):
    for line in text.splitlines():
        if line.startswith(name + "{") and (
            reason is None or f'finished_reason="{reason}"' in line
        ):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def check(args):
    cases = read_workload(args.workload)
    offline = torch.load(args.offline, weights_only=False)
    reference = {r["id"]: r for r in offline["results"]}
    results = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        await wait_ready(client)

        async def rerank(case, path="/rerank", **extra):
            response = await client.post(
                path,
                json={
                    "model": "jina35",
                    "query": case["query"],
                    "documents": case["documents"],
                    **extra,
                },
            )
            response.raise_for_status()
            return response.json()

        for case in cases:
            data = await rerank(case)
            scores = [0.0] * len(case["documents"])
            for row in data["results"]:
                assert row["document"]["text"] == case["documents"][row["index"]]
                scores[row["index"]] = row["relevance_score"]
            error = float(
                np.max(np.abs(np.array(scores) - reference[case["id"]]["scores"]))
            )
            assert error <= 0.01, (case["id"], error)
            assert data["usage"]["prompt_tokens"] == sum(
                map(len, reference[case["id"]]["prompt_token_ids"])
            )
            results.append(
                dict(
                    id=case["id"],
                    max_score_error=error,
                    prompt_tokens=data["usage"]["prompt_tokens"],
                )
            )
        short = cases[0]
        for path in ["/v1/rerank", "/v2/rerank"]:
            data = await rerank(short, path, top_n=1)
            assert len(data["results"]) == 1
        for path in ["/score", "/v1/score"]:
            response = await client.post(
                path,
                json={
                    "model": "jina35",
                    "text_1": short["query"],
                    "text_2": short["documents"],
                },
            )
            response.raise_for_status()
            assert [r["index"] for r in response.json()["data"]] == list(
                range(len(short["documents"]))
            )
            assert (
                np.max(
                    np.abs(
                        np.asarray([row["score"] for row in response.json()["data"]])
                        - reference[short["id"]]["scores"]
                    )
                )
                <= 0.01
            )
        pairs = [short["query"]] * 2
        pair_docs = short["documents"][:2]
        paired = await client.post(
            "/score", json={"text_1": pairs, "text_2": pair_docs}
        )
        paired.raise_for_status()
        for index, doc in enumerate(pair_docs):
            single = await rerank(dict(query=short["query"], documents=[doc]))
            assert (
                abs(
                    paired.json()["data"][index]["score"]
                    - single["results"][0]["relevance_score"]
                )
                < 0.01
            )
        for extra in [
            {"max_tokens_per_doc": 12},
            {"instruction": "other"},
            {"truncate_prompt_tokens": 100},
            {"padding": "max_length"},
            {"top_n": None},
        ]:
            response = await client.post(
                "/rerank", json={"query": "q", "documents": ["d"], **extra}
            )
            assert 400 <= response.status_code < 500
        # A real TCP close after scheduler admission; another client remains valid.
        log_offset = args.server_log.stat().st_size
        disconnect_id = "socket-disconnect-" + uuid.uuid4().hex
        before = (await client.get("/metrics")).text
        abort_before = metric(before, "vllm:request_success_total", "abort")
        target = urlsplit(args.base_url)
        reader, writer = await asyncio.open_connection(
            target.hostname, target.port or 80
        )
        capacity = next(case for case in cases if case["id"] == "capacity")
        body = json.dumps(
            {"query": capacity["query"], "documents": capacity["documents"]}
        ).encode()
        header = (
            f"POST /rerank HTTP/1.1\r\nHost: {target.netloc}\r\n"
            f"Content-Type: application/json\r\nX-Request-Id: {disconnect_id}\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        writer.write(header + body)
        await writer.drain()
        deadline = time.monotonic() + 10
        admitted = False
        while time.monotonic() < deadline:
            observed = (await client.get("/metrics")).text
            if metric(observed, "vllm:num_requests_running") > 0:
                admitted = True
                break
            await asyncio.sleep(0.05)
        assert admitted, "Long request was not observed running before disconnect"
        survivor = asyncio.create_task(rerank(short))
        writer.transport.abort()
        await writer.wait_closed()
        survived = await survivor
        assert len(survived["results"]) == len(short["documents"])
        deadline = time.monotonic() + 10
        abort_after = abort_before
        while time.monotonic() < deadline:
            after = (await client.get("/metrics")).text
            abort_after = metric(after, "vllm:request_success_total", "abort")
            log_text = args.server_log.read_text()[log_offset:]
            abort_lines = [
                line
                for line in log_text.splitlines()
                if "Aborted Jina request" in line and disconnect_id in line
            ]
            if abort_lines and metric(after, "vllm:num_requests_running") == 0:
                break
            await asyncio.sleep(0.1)
        assert abort_lines, "Handler did not confirm engine abort"
        assert metric(after, "vllm:num_requests_running") == 0
        disconnect = dict(
            admitted=admitted,
            abort_count_before=abort_before,
            abort_count_after=abort_after,
            survivor_passed=True,
            abort_log=abort_lines,
            running_after=metric(after, "vllm:num_requests_running"),
        )
    return dict(
        mode=args.mode,
        routes_passed=["/rerank", "/v1/rerank", "/v2/rerank", "/score", "/v1/score"],
        cases=results,
        options_passed=True,
        disconnect=disconnect,
        passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["bf16", "fp8"])
    parser.add_argument("--base-url", default="http://127.0.0.1:18035")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(check(args))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
