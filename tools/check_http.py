"""Exercise real native routes and socket disconnect cleanup."""

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from transformers import AutoTokenizer

from jina_vllm.reranker_v35.config import reranker_block_size
from jina_vllm.reranker_v35.ranking import prepare_request
from tools.bench_serving import served_max_model_len, wait_ready
from tools.compare_reference import model_directory, read_workload


def metric(text, name):
    for line in text.splitlines():
        if line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def check(args):
    cases = read_workload(args.workload)
    tokenizer = AutoTokenizer.from_pretrained(model_directory(), local_files_only=True)

    block_size = reranker_block_size(args.additional_config)
    results = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        await wait_ready(client)
        max_model_len = await served_max_model_len(client)

        def block_lengths(pairs):
            plan = prepare_request(
                pairs, tokenizer, max_model_len, block_size=block_size
            )
            return [len(block.prompt_token_ids) for block in plan.blocks]

        async def completed_blocks():
            response = await client.get("/metrics")
            response.raise_for_status()
            return metric(response.text, "vllm:request_prompt_tokens_count")

        async def rerank(case, path="/rerank", **extra):
            response = await client.post(
                path,
                json={
                    "query": case["query"],
                    "documents": case["documents"],
                    **extra,
                },
            )
            response.raise_for_status()
            return response.json()

        for case in cases:
            before = await completed_blocks()
            data = await rerank(case)
            documents = case["documents"]
            rows = data["results"]
            assert sorted(row["index"] for row in rows) == list(range(len(documents)))
            for row in rows:
                assert row["document"]["text"] == documents[row["index"]]
                assert math.isfinite(row["relevance_score"])
            lengths = block_lengths([(case["query"], documents)])
            assert data["usage"]["prompt_tokens"] == sum(lengths)
            deadline = time.monotonic() + 5
            while (observed := await completed_blocks() - before) < len(lengths):
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.05)
            assert observed == len(lengths), (
                f"Expected {len(lengths)} completed blocks, observed {observed}"
            )
            results.append(
                dict(
                    id=case["id"],
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    block_prompt_tokens=lengths,
                    completed_blocks=int(observed),
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
                    "text_1": short["query"],
                    "text_2": short["documents"],
                },
            )
            response.raise_for_status()
            data = response.json()
            assert [row["index"] for row in data["data"]] == list(
                range(len(short["documents"]))
            )
            assert all(math.isfinite(row["score"]) for row in data["data"])
            assert data["usage"]["prompt_tokens"] == sum(
                block_lengths([(short["query"], short["documents"])])
            )
        pairs = [short["query"]] * 2
        pair_docs = short["documents"][:2]
        paired = await client.post(
            "/score", json={"text_1": pairs, "text_2": pair_docs}
        )
        paired.raise_for_status()
        data = paired.json()
        assert [row["index"] for row in data["data"]] == [0, 1]
        assert all(math.isfinite(row["score"]) for row in data["data"])
        assert data["usage"]["prompt_tokens"] == sum(
            block_lengths(
                [(query, [doc]) for query, doc in zip(pairs, pair_docs, strict=True)]
            )
        )
        for extra in [
            {"max_tokens_per_doc": 12},
            {"instruction": "other"},
            {"chat_template": "other"},
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
        target = urlsplit(args.base_url)
        _, writer = await asyncio.open_connection(target.hostname, target.port or 80)
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
        while time.monotonic() < deadline:
            after = (await client.get("/metrics")).text
            with args.server_log.open("rb") as log:
                log.seek(log_offset)
                log_text = log.read().decode("utf-8", errors="replace")
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
            survivor_passed=True,
            abort_log=abort_lines,
            running_after=metric(after, "vllm:num_requests_running"),
        )
    return dict(
        mode=args.mode,
        block_size=block_size,
        max_model_len=max_model_len,
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
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-config", type=json.loads, default={})
    args = parser.parse_args()
    result = asyncio.run(check(args))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
