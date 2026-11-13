from functools import cache, wraps
from typing import Literal, cast

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.config.cache import register_kv_cache_dtype
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import KVQuantMode

from .contract import ARCH

FP8PrefillMode = Literal["fp8_per_token_head_p_row", "fp8_per_token_head_p_tile"]
FP8_P_ROW = "fp8_per_token_head_p_row"
FP8_P_TILE = "fp8_per_token_head_p_tile"
FP8_PREFILL_MODES = (FP8_P_ROW, FP8_P_TILE)
FP8_PREFILL_BACKEND = "JINA_FP8_PREFILL"


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
    if config.model_config is None or config.model_config.architecture != ARCH:
        raise ValueError(f"kv-cache-dtype={mode} requires {ARCH}")
    if attention.backend_per_kind:
        raise ValueError(f"{FP8_PREFILL_BACKEND} requires one backend for all layers")


class Jina35Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        compilation = vllm_config.compilation_config
        passes = compilation.pass_config
        if passes.fuse_act_quant is None:
            passes.fuse_act_quant = False
        if passes.fuse_norm_quant is None:
            passes.fuse_norm_quant = False
        if passes.enable_qk_norm_rope_fusion is None:
            ops = compilation.custom_ops
            # Mirror the custom RoPE selection used by the pinned vLLM
            # CompilationConfig when the QK pass is explicitly enabled.
            passes.enable_qk_norm_rope_fusion = (
                (current_platform.is_cuda_alike() or current_platform.is_xpu())
                and "-rotary_embedding" not in ops
                and not vllm_config.model_config.enforce_eager
                and compilation.mode in (None, CompilationMode.VLLM_COMPILE)
            )
            if passes.enable_qk_norm_rope_fusion and "+rotary_embedding" not in ops:
                ops.append("+rotary_embedding")
