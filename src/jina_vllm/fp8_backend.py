"""Native vLLM backend selection for pool-free Jina FP8 prefill."""

import torch
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionType

from .config import FP8_P_TILE, FP8_PREFILL_BACKEND, FP8_PREFILL_MODES
from .fp8_prefill import FP8_DEVICE_CAPABILITIES, attend, attention_tile, pack
from .prefill import PrefillMetadata, run_packed_prefill


class JinaFP8PrefillImpl(AttentionImpl[PrefillMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if (num_heads, num_kv_heads, head_size) != (16, 8, 128):
            raise ValueError(
                "JINA_FP8_PREFILL requires 16 Q heads, 8 KV heads, and head size 128"
            )
        if attn_type != AttentionType.DECODER:
            raise ValueError("JINA_FP8_PREFILL requires causal decoder attention")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.window = sliding_window or 0
        self.sliding_window = (sliding_window - 1, 0) if sliding_window else (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype
        self.fp8_tile = attention_tile(torch.cuda.get_device_capability(), self.window)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PrefillMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_packed_prefill(
            self._attend, query, key, value, attn_metadata, output
        )

    def _attend(self, q, k, v, metadata: PrefillMetadata, output):
        attend(
            pack(q, k, v),
            metadata.offsets,
            metadata.max_seq_len,
            output,
            self.window,
            self.kv_cache_dtype == FP8_P_TILE,
            self.fp8_tile,
            self.scale,
        )


class JinaFP8PrefillBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = list(FP8_PREFILL_MODES)

    @staticmethod
    def get_name() -> str:
        return FP8_PREFILL_BACKEND

    @staticmethod
    def get_impl_cls() -> type[JinaFP8PrefillImpl]:
        return JinaFP8PrefillImpl

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("Jina FP8 prefill uses V2 pool-free metadata")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability in FP8_DEVICE_CAPABILITIES
