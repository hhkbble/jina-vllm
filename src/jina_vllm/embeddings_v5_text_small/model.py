from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.jina import JinaEmbeddingsV5DecoderModel
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from jina_vllm.config import fp8_prefill_mode
from jina_vllm.prefill import (
    PrefillModelState,
    fp8_prefill_reason,
    install_pool_free_attention,
    paged_kv_reason,
)
from jina_vllm.rope import validate_rope_config, verify_rotary_cache

logger = init_logger("vllm.jina_vllm.embeddings_v5_text_small")


def validate_model_contract(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_config
    expected = {
        "model_type": "jina_embeddings_v5",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    for field, value in expected.items():
        actual = getattr(config, field, None)
        if actual != value:
            raise ValueError(f"Text Small requires {field}={value!r}, got {actual!r}")
    if (
        not getattr(config, "is_decoder", True)
        or not getattr(config, "is_causal", True)
        or getattr(config, "sliding_window", None) is not None
        or getattr(config, "dual_chunk_attention_config", None)
    ):
        raise ValueError("Text Small requires full causal Qwen3 attention")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Text Small requires tensor and pipeline parallel size 1")
    validate_rope_config(model, 32768, 3500000)


class JinaEmbeddingsV5TextSmall(JinaEmbeddingsV5DecoderModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        validate_model_contract(vllm_config)
        mode = fp8_prefill_mode(vllm_config.cache_config.cache_dtype)
        reason = paged_kv_reason(vllm_config, mode)
        if mode is not None:
            reason = reason or fp8_prefill_reason(vllm_config)
            if reason is not None:
                raise ValueError(f"{mode} requires pool-free prefill: {reason}")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        verify_rotary_cache(vllm_config.model_config, self.model.layers)

        if reason is None:
            attentions = [layer.self_attn for layer in self.model.layers]
            if any(
                not isinstance(layer.attn, Attention)
                or layer.attn.attn_type != AttentionType.DECODER
                or layer.attn.sliding_window is not None
                or layer.attn.head_size_v != layer.attn.head_size
                for layer in attentions
            ):
                raise ValueError(
                    "Text Small requires full causal attention and equal K/V head sizes"
                )
            reason = install_pool_free_attention(attentions, vllm_config, mode)
        self.pool_free_prefill = reason is None
        if self.pool_free_prefill:
            logger.info(
                "Jina Text Small attention mode: pool-free %s", mode or "FA2 BF16"
            )
        else:
            logger.warning("Jina Text Small attention mode: paged KV (%s)", reason)

    def get_model_state_cls(self) -> type[DefaultModelState]:
        return PrefillModelState if self.pool_free_prefill else DefaultModelState
