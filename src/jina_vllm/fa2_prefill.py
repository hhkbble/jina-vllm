"""Native FA2 implementation for the BF16 pool-free path."""

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

from .prefill import PrefillMetadata, run_packed_prefill


class PrefillImpl(FlashAttentionImpl):
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
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=metadata.offsets,
            cu_seqlens_k=metadata.offsets,
            max_seqlen_q=metadata.max_seq_len,
            max_seqlen_k=metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            fa_version=2,
            num_splits=1 if self.batch_invariant_enabled else 0,
        )


class PrefillBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[PrefillImpl]:
        return PrefillImpl
