"""Run one precision at a time; preserve vectors and scores for later comparison."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from jina_vllm.config import FP8_PREFILL_BACKEND, FP8_PREFILL_MODES
from jina_vllm.plugin import PLUGIN_NAME
from jina_vllm.reranker_v35.config import reranker_block_size
from jina_vllm.reranker_v35.contract import (
    ARCH,
    DOC_TOKEN_ID,
    LAYER_TYPES,
    MAX_LENGTH,
    MODEL_ID,
    REVISION,
)
from jina_vllm.reranker_v35.ranking import (
    BlockPlan,
    RankingPlan,
    merge_request,
    prepare_request,
)


def model_directory():
    return os.environ.get("JINA_MODEL_DIR") or snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=True
    )


def engine_kwargs(args):
    options = dict(
        model=model_directory(),
        model_impl="vllm",
        runner="pooling",
        hf_overrides={"architectures": [ARCH], "embedding_size": 512},
        pooler_config={"task": "token_embed"},
        dtype="bfloat16",
        kv_cache_dtype=getattr(args, "kv_cache_dtype", None) or "bfloat16",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        additional_config=args.additional_config,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.batch_tokens,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enable_chunked_prefill=not args.pool_free,
        enable_prefix_caching=False,
        # No FlashInfer GEMM is used by these modes. Avoid unrelated autotuning.
        kernel_config={"enable_flashinfer_autotune": False},
    )
    if args.pool_free:
        options.update(
            max_num_batched_tokens=args.max_model_len,
            max_num_scheduled_tokens=args.max_model_len,
            long_prefill_token_threshold=0,
        )
    if getattr(args, "kv_cache_dtype", None):
        del options["attention_config"]
        options["attention_backend"] = FP8_PREFILL_BACKEND
    if args.mode == "fp8":
        options.update(
            quantization="fp8_per_channel",
            quantization_config={"ignore": ["model.embed_tokens", "projector.*"]},
            kernel_config={
                "enable_flashinfer_autotune": False,
                "linear_backend": "cutlass",
            },
        )
    if args.native_quant_op:
        options["compilation_config"] = {"custom_ops": ["none", "+quant_fp8"]}
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

            def tensor_hash(tensor):
                return hashlib.sha256(
                    tensor.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                ).hexdigest()

            linears.append(
                dict(
                    name=name,
                    method=type(method).__name__,
                    kernel=type(kernel).__name__,
                    dtype=str(module.weight.dtype),
                    shape=list(module.weight.shape),
                    scale_dtype=None if scale is None else str(scale.dtype),
                    input_scale=str(getattr(module, "input_scale", None)),
                    weight_sha256=tensor_hash(module.weight),
                    scale_sha256=None if scale is None else tensor_hash(scale),
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

    os.environ["VLLM_PLUGINS"] = PLUGIN_NAME
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    start = time.perf_counter()
    llm = LLM(**engine_kwargs(args))
    engine_config = llm.llm_engine.vllm_config
    block_size = reranker_block_size(engine_config.additional_config)
    metadata = llm.apply_model(inspect_model)[0]
    if args.pool_free and (
        len(metadata["attention"]) != len(LAYER_TYPES)
        or any(layer["spec"] != "NoneType" for layer in metadata["attention"])
    ):
        raise RuntimeError("Expected pool-free attention on every layer")
    if args.kv_cache_dtype and any(
        layer["backend"] != FP8_PREFILL_BACKEND
        or layer["kv_dtype"] != args.kv_cache_dtype
        for layer in metadata["attention"]
    ):
        raise RuntimeError("Expected the selected FP8 backend and dtype on every layer")
    init_seconds = time.perf_counter() - start
    results = []
    for case in cases:
        start = time.perf_counter()
        plan = prepare_request(
            [(case["query"], case["documents"])],
            llm.get_tokenizer(),
            llm.model_config.max_model_len,
            block_size=block_size,
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
        for block, output in zip(plan.blocks, outputs, strict=True):
            if (
                not output.finished
                or tuple(output.prompt_token_ids) != block.prompt_token_ids
            ):
                raise RuntimeError("Jina block output token IDs or completion mismatch")
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
                block_document_counts=[len(b.output_indices) for b in plan.blocks],
                block_prompt_tokens=[len(b.prompt_token_ids) for b in plan.blocks],
                preprocess_seconds=preprocess,
                inference_seconds=seconds,
            )
        )
        print(f"Completed {args.mode} {case['id']}: {seconds:.3f}s", flush=True)
    return dict(
        mode=args.mode,
        kv_cache_dtype=args.kv_cache_dtype or "bfloat16",
        metadata=metadata,
        block_size=block_size,
        max_model_len=llm.model_config.max_model_len,
        init_seconds=init_seconds,
        results=results,
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
        # Use the reference's own blocks to compare its fusion with CPU fusion.
        blocks = []
        offset = 0
        for prompt_ids in ids:
            end = offset + prompt_ids.count(DOC_TOKEN_ID)
            blocks.append(BlockPlan(0, tuple(range(offset, end)), tuple(prompt_ids)))
            offset = end
        assert offset == len(scores)
        plan = RankingPlan(offset, tuple(blocks))
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
    parser.add_argument("--additional-config", type=json.loads, default={})
    parser.add_argument("--max-model-len", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--batch-tokens", type=int, help="Paged token budget (default: 8192)"
    )
    parser.add_argument(
        "--pool-free",
        action="store_true",
        help="Use pool-free attention with a full-context scheduler budget",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=FP8_PREFILL_MODES,
        help="Select a Jina FP8 prefill mode with --pool-free",
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--native-quant-op",
        action="store_true",
        help="Diagnostic: use native FP8 activation quant custom op",
    )
    parser.add_argument(
        "--hf-attention",
        default="flex_attention",
        choices=["sdpa", "flex_attention", "flash_attention_2"],
    )
    args = parser.parse_args()
    if args.kv_cache_dtype and not args.pool_free:
        parser.error("--kv-cache-dtype requires --pool-free")
    if args.mode == "hf" and (
        args.pool_free or args.additional_config or args.max_model_len != MAX_LENGTH
    ):
        parser.error("HF reference mode uses the checkpoint's own configuration")
    if args.pool_free and (
        args.max_model_len < 1 or args.batch_tokens not in (None, args.max_model_len)
    ):
        parser.error(
            "--pool-free requires a positive --max-model-len and a matching budget"
        )
    if args.batch_tokens is None:
        args.batch_tokens = 8192
    cases = read_workload(args.workload)
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
