"""Compare frozen HF, BF16 and FP8 runs without changing acceptance thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def error_stats(values):
    errors = np.concatenate([np.asarray(x).reshape(-1) for x in values])
    return dict(
        max=float(errors.max()),
        p50=float(np.percentile(errors, 50)),
        p95=float(np.percentile(errors, 95)),
    )


def ranking_metrics(scores, relevance):
    order = np.argsort(-np.asarray(scores), kind="stable")[:10]
    labels = np.asarray(relevance, dtype=float)
    discount = np.log2(np.arange(len(order)) + 2)
    dcg = ((2 ** labels[order] - 1) / discount).sum()
    ideal = ((2 ** np.sort(labels)[::-1][: len(order)] - 1) / discount).sum()
    positives = np.flatnonzero(labels[order] > 0)
    return dict(
        ndcg10=float(dcg / ideal) if ideal else 0.0,
        mrr10=float(1 / (positives[0] + 1)) if len(positives) else 0.0,
    )


def compare(reference, candidate, cases):
    by_id = {row["id"]: row for row in candidate["results"]}
    rows = []
    all_vectors, all_scores = [], []
    for ref in reference["results"]:
        current = by_id[ref["id"]]
        if ref["prompt_token_ids"] != current["prompt_token_ids"]:
            raise RuntimeError(f"Token IDs differ for {ref['id']}")
        vector_errors = [
            (a.float() - b.float()).abs().numpy()
            for a, b in zip(ref["raw"], current["raw"])
        ]
        score_errors = np.abs(np.asarray(ref["scores"]) - current["scores"])
        all_vectors.extend(vector_errors)
        all_scores.append(score_errors)
        k = min(10, len(ref["scores"]))
        a = np.argsort(-np.asarray(ref["scores"]), kind="stable")[:k]
        b = np.argsort(-np.asarray(current["scores"]), kind="stable")[:k]
        row = dict(
            id=ref["id"],
            scores=error_stats([score_errors]),
            vectors=error_stats(vector_errors),
            top10_overlap=len(set(a) & set(b)) / k,
        )
        if ref["id"] in cases:
            case = cases[ref["id"]]
            row["split"] = case["split"]
            row["reference_metrics"] = ranking_metrics(ref["scores"], case["relevance"])
            row["candidate_metrics"] = ranking_metrics(
                current["scores"], case["relevance"]
            )
        rows.append(row)
    quality = {}
    for split in ["calibration", "heldout"]:
        selected = [r for r in rows if r.get("split") == split]
        quality[split] = {}
        for metric in ["ndcg10", "mrr10"]:
            base = float(np.mean([r["reference_metrics"][metric] for r in selected]))
            new = float(np.mean([r["candidate_metrics"][metric] for r in selected]))
            quality[split][metric] = dict(
                reference=base, candidate=new, loss=base - new
            )
    return dict(
        scores=error_stats(all_scores),
        vectors=error_stats(all_vectors),
        mean_top10_overlap=float(np.mean([r["top10_overlap"] for r in rows])),
        quality=quality,
        rows=rows,
    )


def validate_structure(run):
    meta = run["metadata"]
    linears = meta["linears"]
    attention = meta["attention"]
    assert len(linears) == 112
    assert len(attention) == 28
    assert (
        sum(a["window"] == 1024 and a["spec"] == "SlidingWindowSpec" for a in attention)
        == 16
    )
    assert (
        sum(a["window"] is None and a["spec"] == "FullAttentionSpec" for a in attention)
        == 12
    )
    assert all(
        a["attn_type"] == "decoder"
        and a["kv_dtype"] == "bfloat16"
        and a["backend"] == "FLASH_ATTN"
        and a["flash_version"] == 2
        for a in attention
    )
    assert meta["embedding_dtype"] == "torch.bfloat16"
    assert meta["projector_dtypes"] == ["torch.bfloat16", "torch.bfloat16"]
    assert meta["pooler_head"] == "None"
    if run["mode"] == "fp8":
        assert all(
            linear["method"] == "Fp8PtpcOnlineLinearMethod"
            and linear["kernel"] == "CutlassFP8ScaledMMLinearKernel"
            and linear["dtype"] == "torch.float8_e4m3fn"
            and linear["scale_dtype"] == "torch.float32"
            and linear["input_scale"] == "None"
            for linear in linears
        )
    else:
        assert all(
            linear["method"] == "UnquantizedLinearMethod"
            and linear["dtype"] == "torch.bfloat16"
            for linear in linears
        )


def compare_benchmarks(bf16, fp8):
    rows = []
    for base, new in zip(bf16["cells"], fp8["cells"], strict=True):
        assert (base["workload"], base["concurrency"]) == (
            new["workload"],
            new["concurrency"],
        )
        rows.append(
            dict(
                workload=base["workload"],
                concurrency=base["concurrency"],
                bf16_p50_ms=1000 * base["p50_seconds"],
                bf16_p95_ms=1000 * base["p95_seconds"],
                fp8_p50_ms=1000 * new["p50_seconds"],
                fp8_p95_ms=1000 * new["p95_seconds"],
                p50_speedup=base["p50_seconds"] / new["p50_seconds"],
                bf16_requests_per_second=base["requests_per_second"],
                fp8_requests_per_second=new["requests_per_second"],
                bf16_documents_per_second=base["documents_per_second"],
                fp8_documents_per_second=new["documents_per_second"],
                bf16_prompt_tokens_per_second=base["prompt_tokens_per_second"],
                fp8_prompt_tokens_per_second=new["prompt_tokens_per_second"],
            )
        )
    return dict(
        rows=rows,
        geomean_p50_speedup=float(
            np.exp(np.mean(np.log([r["p50_speedup"] for r in rows])))
        ),
        bf16_gpu_peak_device_bytes=max(
            c["gpu_peak_device_bytes"] for c in bf16["cells"]
        ),
        fp8_gpu_peak_device_bytes=max(c["gpu_peak_device_bytes"] for c in fp8["cells"]),
        preprocessing={"bf16": bf16["preprocessing"], "fp8": fp8["preprocessing"]},
    )


def compare_chunked(directory, runs, limit):
    rows = []
    for mode in ["bf16", "fp8"]:
        chunked = torch.load(directory / f"{mode}-chunk2048.pt", weights_only=False)
        base = {row["id"]: row for row in runs[mode]["results"]}
        for row in chunked["results"]:
            original = base[row["id"]]
            assert original["prompt_token_ids"] == row["prompt_token_ids"]
            error = float(
                np.max(np.abs(np.asarray(original["scores"]) - row["scores"]))
            )
            rows.append(dict(mode=mode, id=row["id"], max_score_error=error))
    return dict(
        budgets=[8192, 2048],
        rows=rows,
        passed=all(row["max_score_error"] <= limit for row in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--limits", type=Path, default=Path("acceptance.json"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require HTTP, chunked and benchmark receipts too",
    )
    args = parser.parse_args()
    limits = json.loads(args.limits.read_text())
    cases = {
        r["id"]: r
        for r in map(
            json.loads, (args.directory / "workload.jsonl").read_text().splitlines()
        )
    }
    runs = {
        mode: torch.load(args.directory / f"{mode}.pt", weights_only=False)
        for mode in ["hf", "bf16", "fp8"]
    }
    for mode in ["bf16", "fp8"]:
        validate_structure(runs[mode])
    report = {}
    gates = {}
    for key, ref, new in [("bf16_vs_hf", "hf", "bf16"), ("fp8_vs_bf16", "bf16", "fp8")]:
        result = compare(runs[ref], runs[new], cases)
        report[key] = result
        gate = limits[key]
        gates[key] = dict(
            score=result["scores"]["max"] <= gate["score_max_abs"],
            vector_max=result["vectors"]["max"] <= gate["vector_max_abs"],
            vector_p95=result["vectors"]["p95"] <= gate["vector_p95_abs"],
        )
        if key == "fp8_vs_bf16":
            for metric in ["ndcg10", "mrr10"]:
                gates[key][metric] = (
                    result["quality"]["heldout"][metric]["loss"]
                    <= gate[f"heldout_mean_{metric}_loss"]
                )
    cpu = error_stats(
        [
            np.abs(np.asarray(r["scores"]) - r["cpu_merge_scores"])
            for r in runs["hf"]["results"]
        ]
    )
    report["cpu_merge_vs_hf"] = cpu
    gates["cpu_merge_vs_hf"] = cpu["max"] <= limits["cpu_merge_vs_hf"]["score_max_abs"]
    if args.full:
        benches = {
            mode: json.loads((args.directory / f"bench-{mode}.json").read_text())
            for mode in ["bf16", "fp8"]
        }
        performance = compare_benchmarks(benches["bf16"], benches["fp8"])
        performance["passed"] = (
            performance["geomean_p50_speedup"]
            >= limits["performance"]["minimum_geomean_latency_speedup"]
        )
        (args.directory / "performance-acceptance.json").write_text(
            json.dumps(performance, indent=2) + "\n"
        )
        report["performance"] = performance
        gates["performance"] = performance["passed"]
        chunked = compare_chunked(
            args.directory, runs, limits["chunked_prefill"]["score_max_abs"]
        )
        (args.directory / "chunked-prefill.json").write_text(
            json.dumps(chunked, indent=2) + "\n"
        )
        report["chunked_prefill"] = chunked
        gates["chunked_prefill"] = chunked["passed"]
        report["http"] = {
            mode: json.loads((args.directory / f"http-{mode}.json").read_text())
            for mode in ["bf16", "fp8"]
        }
        gates["http"] = all(receipt["passed"] for receipt in report["http"].values())
    report["gates"] = gates
    report["passed"] = all(
        all(v.values()) if isinstance(v, dict) else v for v in gates.values()
    )
    (args.directory / "numerical-acceptance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if args.full:
        (args.directory / "acceptance-status.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["bf16_vs_hf", "fp8_vs_bf16"]},
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
