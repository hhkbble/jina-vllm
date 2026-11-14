# Native constructor adapted from vLLM jina.py (Apache-2.0).
# Copyright contributors to the vLLM project. See NOTICE.
"""Native Qwen3 computation with v3.5's per-layer attention windows."""

from torch import nn
from transformers import Qwen3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import TokenPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.jina import JinaForRanking, JinaForRankingPool
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from jina_vllm.config import FP8PrefillMode, fp8_prefill_mode
from jina_vllm.prefill import (
    PrefillModelState,
    fp8_prefill_reason,
    paged_kv_layer_reason,
    paged_kv_reason,
    replace_with_pool_free_attention,
)

from .contract import validate_model_contract

logger = init_logger("vllm.jina_vllm.reranker_v35.model")


class RerankerV35DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        index = extract_layer_index(prefix)
        kind = config.layer_types[index]
        window = config.sliding_window if kind == "sliding_attention" else None
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=window,
        )


class RerankerV35Backbone(Qwen3Model):
    decoder_layer_type = RerankerV35DecoderLayer


class JinaRerankerV35(JinaForRanking):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        validate_model_contract(vllm_config)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.projector_dim = config.embedding_size
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.fp8_prefill_mode = fp8_prefill_mode(vllm_config.cache_config.cache_dtype)
        reason = paged_kv_reason(vllm_config, self.fp8_prefill_mode)
        if self.fp8_prefill_mode is not None:
            reason = reason or fp8_prefill_reason(vllm_config)
            if reason is not None:
                raise ValueError(
                    f"{self.fp8_prefill_mode} requires pool-free prefill: {reason}"
                )
        if reason is not None:
            logger.warning("Jina v3.5 attention mode: paged KV (%s)", reason)
        self.model = RerankerV35Backbone(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pool_free_prefill = reason is None and self._enable_pool_free_prefill(
            self.fp8_prefill_mode
        )
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )
        self.pooler = DispatchPooler(
            {"token_embed": TokenPooler(pooling=JinaForRankingPool(self.projector))}
        )

    def get_model_state_cls(self) -> type[DefaultModelState]:
        return PrefillModelState if self.pool_free_prefill else DefaultModelState

    def _enable_pool_free_prefill(self, mode: FP8PrefillMode | None) -> bool:
        attentions = [layer.self_attn for layer in self.model.layers]
        if mode is None:
            reason = paged_kv_layer_reason([module.attn for module in attentions])
            if reason is not None:
                logger.warning("Jina v3.5 attention mode: paged KV (%s)", reason)
                return False
        for module in attentions:
            module.attn = replace_with_pool_free_attention(
                module.attn, self.vllm_config, mode
            )
        logger.info(
            "Jina v3.5 attention mode: pool-free %s (full-prompt prefill)",
            mode or "FA2 BF16",
        )
        return True
