from functools import cache, wraps
from typing import Literal, cast

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.config.cache import register_kv_cache_dtype
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import KVQuantMode

from .embeddings_v5_text_small import ARCH as EMBEDDINGS_V5_TEXT_SMALL_ARCH
from .reranker_v35.contract import ARCH as RERANKER_V35_ARCH

logger = init_logger("vllm.jina_vllm.config")

FP8PrefillMode = Literal["fp8_per_token_head_p_row", "fp8_per_token_head_p_tile"]
FP8_P_ROW = "fp8_per_token_head_p_row"
FP8_P_TILE = "fp8_per_token_head_p_tile"
FP8_PREFILL_MODES = (FP8_P_ROW, FP8_P_TILE)
FP8_PREFILL_BACKEND = "JINA_FP8_PREFILL"
FP8_PREFILL_ARCHITECTURES = frozenset(
    {RERANKER_V35_ARCH, EMBEDDINGS_V5_TEXT_SMALL_ARCH}
)


def configure_large_batch_swiglu(vllm_config: VllmConfig) -> None:
    """Choose SwiGLU before building the MLPs, with the resolved worker config."""
    compilation = vllm_config.compilation_config
    if (
        compilation.mode == CompilationMode.NONE
        or compilation.backend != "inductor"
        or not current_platform.is_cuda()
    ):
        return

    model = vllm_config.model_config
    # The packed gate/up tensor has two intermediate-width halves per token.
    max_int32_rows = (1 << 31) // (2 * model.hf_config.intermediate_size)
    batch_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    if batch_tokens <= max_int32_rows:
        return

    # Inductor can fuse SwiGLU with FP8 quantization using int32 offsets.
    # The native CUDA op uses int64 offsets, including during warmup.
    ops = compilation.custom_ops
    if "-silu_and_mul" in ops:
        raise ValueError(
            f"{model.architecture} CUDA Inductor batches above {max_int32_rows:,}"
            " tokens require the native silu_and_mul op. Remove "
            "'-silu_and_mul' from compilation_config.custom_ops."
        )
    if "all" not in ops and "+silu_and_mul" not in ops:
        ops.append("+silu_and_mul")
        logger.info(
            "Enabled native CUDA SwiGLU for %s with a %d-token batch budget",
            model.architecture,
            batch_tokens,
        )


def restore_fp8_backend() -> AttentionBackendEnum:
    register_prefill_modes()
    return AttentionBackendEnum[FP8_PREFILL_BACKEND]


@cache
def register_prefill_modes() -> None:
    register_backend(FP8_PREFILL_BACKEND, "jina_vllm.fp8_backend.JinaFP8PrefillBackend")

    for mode in FP8_PREFILL_MODES:

        @register_kv_cache_dtype(mode)
        class PoolFreeCacheDType:
            name = mode

            def torch_dtype(self) -> torch.dtype:
                # The runner still resolves an empty KV placeholder's dtype.
                return torch.bfloat16

            def quant_mode(self) -> KVQuantMode:
                return KVQuantMode.NONE

    # Spawned engine processes unpickle the config before loading plugins.
    reduce_backend = AttentionBackendEnum.__reduce_ex__

    def reduce(self, protocol):
        if self.name == FP8_PREFILL_BACKEND:
            return restore_fp8_backend, ()
        return reduce_backend(self, protocol)

    AttentionBackendEnum.__reduce_ex__ = reduce

    # These registries are process-wide. Validate the pair and its architecture
    # at config construction, even for models that bypass backend selection.
    verify = VllmConfig.try_verify_and_update_config

    @wraps(verify)
    def verify_prefill_config(config: VllmConfig) -> None:
        validate_prefill_selection(config)
        verify(config)

    VllmConfig.try_verify_and_update_config = verify_prefill_config


def fp8_prefill_mode(cache_dtype: str) -> FP8PrefillMode | None:
    return (
        cast(FP8PrefillMode, cache_dtype) if cache_dtype in FP8_PREFILL_MODES else None
    )


def validate_prefill_selection(config: VllmConfig) -> None:
    mode = fp8_prefill_mode(config.cache_config.cache_dtype)
    attention = config.attention_config
    backend = AttentionBackendEnum[FP8_PREFILL_BACKEND]
    if attention.backend != backend:
        if backend in attention.backend_per_kind.values():
            raise ValueError(f"Select {FP8_PREFILL_BACKEND} as the global backend")
        if mode is not None:
            raise ValueError(f"kv-cache-dtype={mode} requires {FP8_PREFILL_BACKEND}")
        return
    if mode is None:
        choices = " or ".join(FP8_PREFILL_MODES)
        raise ValueError(f"{FP8_PREFILL_BACKEND} requires kv-cache-dtype {choices}")
    if (
        config.model_config is None
        or config.model_config.architecture not in FP8_PREFILL_ARCHITECTURES
    ):
        architectures = ", ".join(sorted(FP8_PREFILL_ARCHITECTURES))
        raise ValueError(f"kv-cache-dtype={mode} requires {architectures}")
    if attention.backend_per_kind:
        raise ValueError(f"{FP8_PREFILL_BACKEND} requires one backend for all layers")
