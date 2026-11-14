"""Check Text Small's checkpoint RoPE and YaRN through native offline pooling."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from vllm import LLM, PoolingParams

from jina_vllm.embeddings_v5_text_small import ARCH
from jina_vllm.plugin import PLUGIN_NAME

NATIVE_LENGTH = 32768


def inspect_rope(model):
    return {
        "rope_parameters": model.vllm_config.model_config.hf_config.rope_parameters,
        "cache_rows": [
            layer.self_attn.rotary_emb.cos_sin_cache.shape[0]
            for layer in model.model.layers
        ],
        "attention": [
            (
                layer.self_attn.attn.get_attn_backend().get_name(),
                layer.self_attn.attn.impl.vllm_flash_attn_version,
            )
            for layer in model.model.layers
        ],
        "pool_free": model.pool_free_prefill,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("default", "yarn", "extended"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["VLLM_PLUGINS"] = PLUGIN_NAME
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    model = os.environ.get("JINA_EMBED_MODEL_DIR") or snapshot_download(
        "jinaai/jina-embeddings-v5-text-small",
        revision="dd76d535f5447ca3897a9c893fb1e612ead98192",
        local_files_only=True,
    )
    overrides = {"architectures": [ARCH], "jina_task": "retrieval"}
    if args.mode != "default":
        overrides["rope_parameters"] = {
            "rope_type": "yarn",
            "rope_theta": 3500000,
            "original_max_position_embeddings": NATIVE_LENGTH,
            "factor": 2.0,
        }
    limit = NATIVE_LENGTH
    if args.mode == "extended":
        overrides["max_position_embeddings"] = NATIVE_LENGTH * 2
        limit += 1

    llm = LLM(
        model=model,
        model_impl="vllm",
        trust_remote_code=True,
        runner="pooling",
        hf_overrides=overrides,
        pooler_config={"task": "embed", "seq_pooling_type": "LAST"},
        dtype="bfloat16",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        kv_cache_dtype="bfloat16",
        max_model_len=limit,
        max_num_batched_tokens=limit,
        max_num_scheduled_tokens=limit,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        kernel_config={"enable_flashinfer_autotune": False},
    )
    metadata = llm.apply_model(inspect_rope)[0]
    assert metadata["pool_free"] and set(metadata["attention"]) == {("FLASH_ATTN", 2)}
    assert (
        metadata["cache_rows"]
        == [NATIVE_LENGTH if args.mode == "default" else NATIVE_LENGTH * 2] * 28
    )
    tokenizer = llm.get_tokenizer()
    prefix = tokenizer.encode("Document: ", add_special_tokens=False)
    fill = tokenizer.encode(
        " Ocean tides depend on the gravitational pull of the Moon. "
        " Wind transfers energy to surface waves.",
        add_special_tokens=False,
    )
    length = limit if args.mode == "extended" else 4096
    prompt = prefix + (fill * (length // len(fill) + 1))[: length - len(prefix)]
    (output,) = llm.embed(
        [{"prompt_token_ids": prompt}],
        pooling_params=PoolingParams(),
        use_tqdm=False,
    )
    assert output.finished and output.prompt_token_ids == prompt
    vector = torch.tensor(output.outputs.embedding)
    assert vector.shape == (1024,) and torch.isfinite(vector).all()
    torch.testing.assert_close(vector.norm(), torch.tensor(1.0))
    result = {
        "mode": args.mode,
        "max_model_len": llm.model_config.max_model_len,
        "prompt_tokens": length,
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "worker": metadata,
        "embedding": vector.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
