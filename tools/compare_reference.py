"""Run one precision at a time; preserve vectors and scores for later comparison."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from jina_vllm.contract import MODEL_ID, REVISION
from jina_vllm.ranking import merge_request, prepare_request


def model_directory():
    return os.environ.get("JINA_MODEL_DIR") or snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=True
    )


def engine_kwargs(args):
    options = dict(
        model=model_directory(),
        model_impl="vllm",
        runner="pooling",
        hf_overrides={"architectures": ["JinaRerankerV35"], "embedding_size": 512},
        pooler_config={"task": "token_embed"},
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=131072,
        max_num_batched_tokens=args.batch_tokens,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        # No FlashInfer GEMM is used by these modes. Avoid unrelated autotuning.
        kernel_config={"enable_flashinfer_autotune": False},
    )
    if args.mode == "fp8":
        options.update(
            quantization="fp8_per_channel",
            quantization_config={"ignore": ["model.embed_tokens", "projector.*"]},
            kernel_config={
                "enable_flashinfer_autotune": False,
                "linear_backend": "cutlass",
            },
        )
    if args.profile:
        options["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile.resolve()),
        }
    return options


def inspect_model(model):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.linear import LinearBase

    linears, attention = [], []
    for name, module in model.named_modules():
        if isinstance(module, LinearBase):
            method = module.quant_method
            kernel = getattr(method, "fp8_linear", None)
            scale = getattr(module, "weight_scale", None)
            linears.append(
                dict(
                    name=name,
                    method=type(method).__name__,
                    kernel=type(kernel).__name__,
                    dtype=str(module.weight.dtype),
                    shape=list(module.weight.shape),
                    scale_dtype=None if scale is None else str(scale.dtype),
                    input_scale=str(getattr(module, "input_scale", None)),
                )
            )
        if isinstance(module, Attention):
            spec = module.get_kv_cache_spec(model.vllm_config)
            attention.append(
                dict(
                    name=name,
                    window=module.sliding_window,
                    attn_type=module.attn_type,
                    kv_dtype=module.kv_cache_dtype,
                    spec=type(spec).__name__,
                    backend=module.attn_backend.get_name(),
                    flash_version=getattr(module.impl, "vllm_flash_attn_version", None),
                )
            )
    return dict(
        linears=linears,
        attention=attention,
        embedding_dtype=str(model.model.embed_tokens.weight.dtype),
        projector_dtypes=[str(p.dtype) for p in model.projector.parameters()],
        pooler_head=str(model.pooler.poolers_by_task["token_embed"].head),
    )


def read_workload(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def native_run(args, cases):
    from vllm import LLM, PoolingParams

    os.environ["VLLM_PLUGINS"] = "jina_reranker_v35"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    start = time.perf_counter()
    llm = LLM(**engine_kwargs(args))
    metadata = llm.apply_model(inspect_model)[0]
    init_seconds = time.perf_counter() - start
    results = []
    for case in cases:
        start = time.perf_counter()
        plan = prepare_request(
            [(case["query"], case["documents"])], llm.get_tokenizer(), 131072
        )
        preprocess = time.perf_counter() - start
        if args.profile and not results:
            llm.start_profile()
        start = time.perf_counter()
        outputs = llm.encode(
            [{"prompt_token_ids": list(b.prompt_token_ids)} for b in plan.blocks],
            pooling_task="token_embed",
            pooling_params=PoolingParams(task="token_embed"),
            tokenization_kwargs={"add_special_tokens": False},
            use_tqdm=False,
        )
        raw = [out.outputs.data for out in outputs]
        scores = merge_request(plan, raw)
        seconds = time.perf_counter() - start
        if args.profile and not results:
            llm.stop_profile()
        results.append(
            dict(
                id=case["id"],
                scores=scores,
                raw=raw,
                prompt_token_ids=[list(b.prompt_token_ids) for b in plan.blocks],
                preprocess_seconds=preprocess,
                inference_seconds=seconds,
            )
        )
        print(f"Completed {args.mode} {case['id']}: {seconds:.3f}s", flush=True)
    return dict(
        mode=args.mode, metadata=metadata, init_seconds=init_seconds, results=results
    )


def hf_run(args, cases):
    from transformers import AutoModel, AutoTokenizer

    directory = model_directory()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    start = time.perf_counter()
    model = (
        AutoModel.from_pretrained(
            directory,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=args.hf_attention,
        )
        .eval()
        .cuda()
    )
    model._tokenizer = tokenizer
    original_forward = model.forward
    raw = []
    ids = []

    def capture(*pos, **kwargs):
        output = original_forward(*pos, **kwargs)
        raw.append(
            torch.cat([output.doc_embeds[0], output.query_embeds[0]], dim=0).cpu()
        )
        ids.append(kwargs["input_ids"][0].cpu().tolist())
        return output

    model.forward = capture
    init_seconds = time.perf_counter() - start
    results = []
    for case in cases:
        raw.clear()
        ids.clear()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            ranked = model.rerank(case["query"], case["documents"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        scores = [0.0] * len(ranked)
        for item in ranked:
            scores[int(item["index"])] = float(item["relevance_score"])
        plan = prepare_request([(case["query"], case["documents"])], tokenizer, 131072)
        assert ids == [list(b.prompt_token_ids) for b in plan.blocks]
        results.append(
            dict(
                id=case["id"],
                scores=scores,
                raw=list(raw),
                prompt_token_ids=list(ids),
                cpu_merge_scores=merge_request(plan, raw),
                inference_seconds=seconds,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        print(f"Completed HF {case['id']}: {seconds:.3f}s", flush=True)
    return dict(
        mode="hf",
        attention=args.hf_attention,
        init_seconds=init_seconds,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hf", "bf16", "fp8"], required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--hf-attention",
        default="sdpa",
        choices=["sdpa", "flex_attention", "flash_attention_2"],
    )
    parser.add_argument("--split", choices=["calibration", "heldout"])
    args = parser.parse_args()
    cases = read_workload(args.workload)
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    result = (hf_run if args.mode == "hf" else native_run)(args, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    summary = {k: v for k, v in result.items() if k != "results"}
    summary["results"] = [
        {k: v for k, v in row.items() if k not in ("raw", "prompt_token_ids")}
        for row in result["results"]
    ]
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
