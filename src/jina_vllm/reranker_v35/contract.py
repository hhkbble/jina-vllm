from importlib.metadata import version

ARCH = "JinaRerankerV35"
MODEL_ID = "jinaai/jina-reranker-v3.5"
REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
MAX_LENGTH = 131072
INTERMEDIATE_SIZE = 3072
DEFAULT_BLOCK_SIZE = 125
DOC_MARKER = "<|embed_token|>"
QUERY_MARKER = "<|rerank_token|>"
DOC_TOKEN_ID = 151670
QUERY_TOKEN_ID = 151671
FULL_LAYERS = {1, 2, 6, 7, 11, 12, 16, 17, 21, 22, 26, 27}
LAYER_TYPES = tuple(
    "full_attention" if i in FULL_LAYERS else "sliding_attention" for i in range(28)
)


def validate_block_size(block_size: int) -> None:
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Jina reranker_v35.block_size must be a positive integer")


def validate_version() -> None:
    from packaging.version import Version

    if Version(version("vllm")).public != "0.33.0":
        raise RuntimeError("Jina v3.5 requires vLLM 0.33.0")


def validate_tokenizer(tokenizer) -> None:
    expected = {
        "model_max_length": MAX_LENGTH,
        "truncation_side": "right",
    }
    for field, value in expected.items():
        if getattr(tokenizer, field, None) != value:
            raise ValueError(f"Tokenizer {field} must be {value!r}")
    for marker, token_id in (
        (DOC_MARKER, DOC_TOKEN_ID),
        (QUERY_MARKER, QUERY_TOKEN_ID),
    ):
        if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Tokenizer marker {marker} must encode as [{token_id}]")


def validate_model_contract(vllm_config) -> None:
    validate_version()
    model = vllm_config.model_config
    config = model.hf_config
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "embedding_size": 512,
        "sliding_window": 1024,
        "max_window_layers": 28,
        "hidden_act": "silu",
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
    }
    errors = [
        f"{key}={getattr(config, key, None)!r}, expected {value!r}"
        for key, value in expected.items()
        if getattr(config, key, None) != value
    ]
    if tuple(config.layer_types) != LAYER_TYPES:
        errors.append("layer_types must match the official 16 sliding / 12 full layers")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        errors.append("tensor_parallel_size and pipeline_parallel_size must be 1")
    if (
        model.disable_sliding_window
        or vllm_config.cache_config.sliding_window is not None
    ):
        errors.append("mixed attention requires per-layer windows and sliding enabled")
    if not getattr(config, "is_causal", True):
        errors.append("causal attention is required")
    if errors:
        raise ValueError("Invalid Jina v3.5 model contract: " + "; ".join(errors))

    from jina_vllm.rope import validate_rope_config

    validate_rope_config(model, MAX_LENGTH, 1000000)
