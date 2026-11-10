# Native constructor adapted from vLLM jina.py (Apache-2.0).
# Copyright contributors to the vLLM project. See NOTICE.
"""Native Qwen3 computation with v3.5's per-layer attention windows."""

from torch import nn
from transformers import Qwen3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import TokenPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.jina import JinaForRanking, JinaForRankingPool
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix

from .contract import validate_model_contract


class Jina35DecoderLayer(Qwen3DecoderLayer):
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


class Jina35Backbone(Qwen3Model):
    decoder_layer_type = Jina35DecoderLayer


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
        self.model = Jina35Backbone(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )
        self.pooler = DispatchPooler(
            {"token_embed": TokenPooler(pooling=JinaForRankingPool(self.projector))}
        )
