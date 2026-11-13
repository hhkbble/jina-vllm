"""Causal FA2 over packed, complete prompts on vLLM Model Runner V2."""

from dataclasses import dataclass

import torch
from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger("vllm.jina_vllm.prefill")


def paged_kv_reason(config: VllmConfig) -> str | None:
    model = config.model_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    attention = config.attention_config
    parallel = config.parallel_config
    if not (current_platform.is_cuda() and config.use_v2_model_runner):
        return "NVIDIA CUDA and Model Runner V2 are required"
    if model.runner_type != "pooling":
        return "the runner is not pooling"
    if cache.cache_dtype not in ("auto", "bfloat16"):
        return f"kv-cache-dtype={cache.cache_dtype} was selected"
    if attention.backend != AttentionBackendEnum.FLASH_ATTN or (
        attention.flash_attn_version != 2
    ):
        return "FLASH_ATTN and flash_attn_version=2 must both be explicit"
    if attention.use_non_causal:
        return "non-causal attention was requested"
    if model.dtype != torch.bfloat16:
        return "Q/K/V must be BF16"
    if scheduler.enable_chunked_prefill:
        return "chunked prefill is enabled"
    if cache.enable_prefix_caching:
        return "prefix caching is enabled"
    if model.original_max_model_len == -1:
        return "max-model-len=-1 requests native KV-based context auto-fit"
    max_len = model.max_model_len
    batch_budget = scheduler.max_num_batched_tokens
    schedule_budget = scheduler.max_num_scheduled_tokens or batch_budget
    if min(batch_budget, schedule_budget) < max_len:
        return (
            f"token budgets (batched={batch_budget}, scheduled={schedule_budget}) "
            f"must cover max-model-len={max_len}"
        )
    threshold = scheduler.long_prefill_token_threshold
    if 0 < threshold < max_len:
        return f"long-prefill-token-threshold={threshold} can split mixed batches"
    if scheduler.scheduler_cls is not None or config.speculative_config is not None:
        return "a custom scheduler or speculative decoding was configured"
    if (
        config.kv_transfer_config is not None
        or attention.hisparse_config is not None
        or cache.kv_cache_memory_bytes is not None
        or cache.num_gpu_blocks_override is not None
        or cache.kv_sharing_fast_prefill
    ):
        return "a native KV transfer, allocation, or sharing option was configured"
    if (
        parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or (parallel.use_ubatching and parallel.data_parallel_size > 1)
    ):
        return "context parallelism or microbatching was configured"
    compilation = config.compilation_config
    if compilation.cudagraph_mode.has_full_cudagraphs() or (
        compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE
        and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        and "vllm::unified_attention_with_output"
        not in (compilation.splitting_ops or ())
    ):
        return "CUDA graph configuration requires capturing the attention kernel"
    return None


def paged_kv_layer_reason(layers: list[Attention]) -> str | None:
    if any(
        layer.get_attn_backend() is not FlashAttentionBackend
        or type(layer.impl) is not FlashAttentionImpl
        or layer.impl.vllm_flash_attn_version != 2
        for layer in layers
    ):
        return "at least one attention layer did not resolve to native FA2"
    if any(
        layer.dtype != torch.bfloat16 or layer.kv_cache_torch_dtype != torch.bfloat16
        for layer in layers
    ):
        return "at least one attention layer did not resolve to BF16 Q/K/V and KV"
    return None


@dataclass(frozen=True)
class PrefillMetadata:
    offsets: torch.Tensor
    num_tokens: int
    max_seq_len: int


class PrefillModelState(DefaultModelState):
    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str]:
        return AttentionCGSupport.NEVER, "Jina v3.5 pool-free FA2"

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        self.layer_names = tuple(layers)
        if (
            kv_cache_config.kv_cache_groups
            or kv_cache_config.kv_cache_tensors
            or any(layer.kv_cache.numel() for layer in layers.values())
        ):
            raise RuntimeError("Jina v3.5 pool-free prefill received a KV pool")
        logger.info(
            "Jina v3.5 pool-free FA2 allocation: kv_cache_tensors=0, "
            "kv_cache_groups=0, layer_kv_elements=0 (%d layers)",
            len(layers),
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        # Called only for real requests, including the native pooling warmup.
        if (
            input_batch.num_computed_tokens_np.any()
            or (input_batch.num_scheduled_tokens != input_batch.prefill_len_np).any()
        ):
            raise RuntimeError("Jina v3.5 pool-free FA2 requires complete prompts")

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, PrefillMetadata]:
        metadata = PrefillMetadata(
            offsets=input_batch.query_start_loc[: input_batch.num_reqs + 1],
            num_tokens=input_batch.num_tokens,
            max_seq_len=int(input_batch.num_scheduled_tokens.max()),
        )
        return dict.fromkeys(self.layer_names, metadata)


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
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        if attn_metadata is None:
            return output.zero_()  # Native profile may skip attention metadata.
        num_tokens = attn_metadata.num_tokens
        flash_attn_varlen_func(
            q=query[:num_tokens],
            k=key[:num_tokens],
            v=value[:num_tokens],
            out=output[:num_tokens],
            cu_seqlens_q=attn_metadata.offsets,
            cu_seqlens_k=attn_metadata.offsets,
            max_seqlen_q=attn_metadata.max_seq_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            fa_version=2,
            num_splits=1 if self.batch_invariant_enabled else 0,
        )
        # Piecewise graphs can pad the backbone beyond the real FA2 rows.
        if num_tokens < output.shape[0]:
            output[num_tokens:].zero_()
        return output


class PrefillBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[PrefillImpl]:
        return PrefillImpl


class PrefillAttention(Attention):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None


def replace_with_pool_free_attention(layer: Attention, config: VllmConfig) -> Attention:
    # The native layer already selected and validated its backend. Replace only
    # its attention core, keeping Qwen3 projections and the native compiled ops.
    # The native KV update op is a no-op when a layer has no slot mapping.
    del config.compilation_config.static_forward_context[layer.layer_name]
    return PrefillAttention(
        layer.num_heads,
        layer.head_size,
        layer.impl.scale,
        num_kv_heads=layer.num_kv_heads,
        cache_config=config.cache_config,
        quant_config=layer.quant_config,
        per_layer_sliding_window=layer.sliding_window,
        prefix=layer.layer_name,
        attn_backend=PrefillBackend,
    )
