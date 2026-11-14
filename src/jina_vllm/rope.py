import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

logger = init_logger("vllm.jina_vllm.rope")


def validate_rope_config(model: "ModelConfig", native_length: int, theta: int) -> None:
    config = model.hf_config
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict) or rope.get("rope_theta") != theta:
        raise ValueError(f"Jina requires flat rope_parameters with rope_theta={theta}")

    rope_type = rope.get("rope_type")
    positions = getattr(config, "max_position_embeddings", None)
    if type(positions) is not int:
        raise ValueError("Jina max_position_embeddings must be an integer")
    if rope_type == "default":
        if rope.keys() != {"rope_type", "rope_theta"} or positions != native_length:
            raise ValueError(
                f"Jina default RoPE requires max_position_embeddings={native_length} "
                "and the checkpoint's rope_parameters"
            )
    elif rope_type == "yarn":
        if rope.keys() != {
            "rope_type",
            "rope_theta",
            "original_max_position_embeddings",
            "factor",
        }:
            raise ValueError(
                "Jina YaRN requires rope_type, rope_theta, "
                "original_max_position_embeddings, and factor"
            )
        original = rope["original_max_position_embeddings"]
        if type(original) is not int or original != native_length:
            raise ValueError(
                f"Jina YaRN requires original_max_position_embeddings={native_length}"
            )
        factor = rope["factor"]
        if type(factor) not in (int, float) or factor <= 1:
            raise ValueError("Jina YaRN factor must be finite and greater than 1")
        try:
            scaled_positions = original * float(factor)
        except OverflowError:
            scaled_positions = math.inf
        if not math.isfinite(scaled_positions):
            raise ValueError("Jina YaRN factor must yield a finite context length")
        capacity = math.ceil(scaled_positions)
        if not native_length <= positions <= capacity:
            raise ValueError(
                f"Jina YaRN max_position_embeddings={positions!r} must be between "
                f"{native_length} and {capacity}"
            )
    else:
        raise ValueError(
            f"Jina RoPE requires rope_type=default or yarn, got {rope_type!r}"
        )

    if model.max_model_len > positions:
        raise ValueError(
            f"Jina max_model_len={model.max_model_len} exceeds "
            f"max_position_embeddings={positions}"
        )


def verify_rotary_cache(
    model: "ModelConfig", layers: Iterable["Qwen3DecoderLayer"]
) -> None:
    rows = min(layer.self_attn.rotary_emb.cos_sin_cache.shape[0] for layer in layers)
    if rows < model.max_model_len:
        raise ValueError(
            f"Jina rotary cache has {rows} positions for "
            f"max_model_len={model.max_model_len}"
        )
    logger.info(
        "%s rotary cache: %s, rows=%d, max_model_len=%d",
        model.architecture,
        model.hf_config.rope_parameters,
        rows,
        model.max_model_len,
    )
