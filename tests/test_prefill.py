from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationMode,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.cache import get_kv_cache_dtype_handler
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVQuantMode

from jina_vllm.config import (
    fp8_prefill_mode,
    register_prefill_modes,
    validate_prefill_selection,
)
from jina_vllm.embeddings_v5_text_small import ARCH as EMBEDDINGS_ARCH
from jina_vllm.fp8_backend import JinaFP8PrefillImpl
from jina_vllm.prefill import (
    PrefillModelState,
    fp8_prefill_reason,
    paged_kv_layer_reason,
    paged_kv_reason,
)
from jina_vllm.reranker_v35.config import RerankerV35Config


@pytest.fixture
def prefill_config(monkeypatch):
    monkeypatch.setattr("jina_vllm.prefill.current_platform.is_cuda", lambda: True)
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            runner_type="pooling",
            max_model_len=16384,
            original_max_model_len=16384,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=16384,
            is_encoder_decoder=False,
            runner_type="pooling",
            max_num_batched_tokens=16384,
            enable_chunked_prefill=False,
        ),
        cache_config=CacheConfig(cache_dtype="auto", enable_prefix_caching=False),
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASH_ATTN, flash_attn_version=2
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_pool_free_accepts_effective_full_prompt_budget(prefill_config):
    scheduler = prefill_config.scheduler_config
    scheduler.max_num_scheduled_tokens = 16384
    scheduler.long_prefill_token_threshold = 16384
    assert paged_kv_reason(prefill_config) is None


@pytest.mark.parametrize(
    "section,field,value,reason",
    [
        ("", "use_v2_model_runner", False, "Runner V2"),
        ("attention_config", "backend", None, "explicit"),
        ("attention_config", "flash_attn_version", None, "explicit"),
        ("cache_config", "cache_dtype", "fp8_e4m3", "kv-cache-dtype"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("model_config", "dtype", torch.float16, "BF16"),
        ("scheduler_config", "enable_chunked_prefill", True, "chunked prefill"),
        ("scheduler_config", "max_num_batched_tokens", 8192, "batched=8192"),
        ("scheduler_config", "max_num_scheduled_tokens", 8192, "scheduled=8192"),
        ("scheduler_config", "long_prefill_token_threshold", 8192, "can split"),
    ],
)
def test_pool_free_falls_back_without_changing_config(
    prefill_config, section, field, value, reason
):
    config = getattr(prefill_config, section) if section else prefill_config
    setattr(config, field, value)
    assert reason in paged_kv_reason(prefill_config)
    assert getattr(config, field) == value


def test_pool_free_checks_every_resolved_layer():
    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 2
    layer = SimpleNamespace(
        get_attn_backend=lambda: FlashAttentionBackend,
        impl=impl,
        dtype=torch.bfloat16,
        kv_cache_torch_dtype=torch.bfloat16,
    )
    assert paged_kv_layer_reason([layer]) is None
    other_impl = object.__new__(FlashAttentionImpl)
    other_impl.vllm_flash_attn_version = 4
    other = SimpleNamespace(**vars(layer))
    other.impl = other_impl
    assert "native FA2" in paged_kv_layer_reason([layer, other])


@pytest.mark.parametrize(
    "mode", ["fp8_per_token_head_p_row", "fp8_per_token_head_p_tile"]
)
def test_explicit_fp8_config_and_fusion_defaults(monkeypatch, mode):
    register_prefill_modes()
    cache = CacheConfig(cache_dtype=mode)
    handler = get_kv_cache_dtype_handler(mode)
    assert handler.quant_mode() == KVQuantMode.NONE
    assert handler.torch_dtype() == torch.bfloat16
    assert fp8_prefill_mode(cache.cache_dtype) == mode
    monkeypatch.setattr(
        "jina_vllm.reranker_v35.config.current_platform.is_cuda_alike", lambda: True
    )
    compilation = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, pass_config={"fuse_norm_quant": True}
    )
    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16_384),
    )
    RerankerV35Config.verify_and_update_config(config)
    assert compilation.pass_config.enable_qk_norm_rope_fusion is True
    assert compilation.pass_config.fuse_act_quant is False
    assert compilation.pass_config.fuse_norm_quant is True
    assert "+rotary_embedding" in compilation.custom_ops


@pytest.mark.parametrize(
    "backend,dtype,architecture,error",
    [
        ("JINA_FP8_PREFILL", "bfloat16", "JinaRerankerV35", "requires kv-cache-dtype"),
        ("FLASH_ATTN", "fp8_per_token_head_p_row", "JinaRerankerV35", "requires JINA"),
        (
            "JINA_FP8_PREFILL",
            "fp8_per_token_head_p_tile",
            "Other",
            "requires .*JinaRerankerV35",
        ),
        (
            "JINA_FP8_PREFILL",
            "fp8_per_token_head_p_tile",
            "JinaEmbeddingsV5Model",
            "requires .*JinaEmbeddingsV5TextSmall",
        ),
    ],
)
def test_registered_fp8_selection_rejects_invalid_pair(
    backend, dtype, architecture, error
):
    register_prefill_modes()
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=architecture),
        cache_config=CacheConfig(cache_dtype=dtype),
        attention_config=AttentionConfig(backend=backend),
    )
    with pytest.raises(ValueError, match=error):
        VllmConfig.try_verify_and_update_config(config)


@pytest.mark.parametrize(
    "mode", ["fp8_per_token_head_p_row", "fp8_per_token_head_p_tile"]
)
def test_text_small_architecture_accepts_explicit_fp8_prefill(mode):
    register_prefill_modes()
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=EMBEDDINGS_ARCH),
        cache_config=CacheConfig(cache_dtype=mode),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.JINA_FP8_PREFILL),
    )
    validate_prefill_selection(config)


@pytest.mark.parametrize(
    "heads,kv_heads,dim,attn_type,error",
    [
        (8, 8, 128, "decoder", "16 Q heads"),
        (16, 4, 128, "decoder", "8 KV heads"),
        (16, 8, 64, "decoder", "head size 128"),
        (16, 8, 128, "encoder_only", "causal decoder"),
    ],
)
def test_fp8_impl_rejects_incompatible_attention(
    heads, kv_heads, dim, attn_type, error
):
    with pytest.raises(ValueError, match=error):
        JinaFP8PrefillImpl(
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_size=dim,
            attn_type=attn_type,
            scale=dim**-0.5,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_per_token_head_p_row",
        )


def test_explicitly_disabled_qk_fusion_and_missing_fp8_option(monkeypatch):
    monkeypatch.setattr(
        "jina_vllm.reranker_v35.config.current_platform.is_cuda_alike", lambda: True
    )
    compilation = CompilationConfig(
        mode=CompilationMode.NONE,
        pass_config={"enable_qk_norm_rope_fusion": False, "fuse_act_quant": True},
    )
    config = SimpleNamespace(additional_config={}, compilation_config=compilation)
    RerankerV35Config.verify_and_update_config(config)
    assert fp8_prefill_mode("fp8_per_token_head") is None
    assert compilation.pass_config.enable_qk_norm_rope_fusion is False
    assert compilation.pass_config.fuse_act_quant is True
    assert compilation.pass_config.fuse_norm_quant is False
    assert "+rotary_embedding" not in compilation.custom_ops


def test_fp8_prefill_checks_device_and_linear_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    capability = (8, 0)
    monkeypatch.setattr(
        "jina_vllm.prefill.current_platform.get_device_capability",
        lambda device_id: capability if device_id == 3 else None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(quantization=None),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        cache_config=CacheConfig(),
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=None, max_num_batched_tokens=16384
        ),
    )
    assert "FP8 MMA" in fp8_prefill_reason(config)
    capability = (8, 9)
    assert fp8_prefill_reason(config) is None
    config.model_config.quantization = "fp8_per_channel"
    assert "CUTLASS" in fp8_prefill_reason(config)
    config.kernel_config.linear_backend = "cutlass"
    assert fp8_prefill_reason(config) is None


@pytest.mark.parametrize("scheduled", [None, 524289])
def test_fp8_prefill_caps_effective_token_budget(prefill_config, scheduled):
    scheduler = prefill_config.scheduler_config
    scheduler.max_num_batched_tokens = 524289
    scheduler.max_num_scheduled_tokens = scheduled
    assert "524289 exceeds" in fp8_prefill_reason(prefill_config)


def test_model_state_uses_only_real_sequence_boundaries():
    state = object.__new__(PrefillModelState)
    state.layer_names = ("sliding", "full")
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=11,
        num_tokens_after_padding=16,
        query_start_loc=torch.tensor([0, 3, 11, 11], dtype=torch.int32),
        num_scheduled_tokens=np.array([3, 8]),
        prefill_len_np=np.array([3, 8]),
        num_computed_tokens_np=np.array([0, 0]),
    )
    state.preprocess_state(batch, (), None, None)
    by_layer = state.prepare_attn(batch, CUDAGraphMode.PIECEWISE, (), None, [], None)
    metadata = by_layer["sliding"]
    assert by_layer["full"] is metadata
    assert metadata.offsets.tolist() == [0, 3, 11]
    assert (metadata.num_tokens, metadata.max_seq_len) == (11, 8)

    batch.prefill_len_np[1] = 9
    with pytest.raises(RuntimeError, match="requires complete prompts"):
        state.preprocess_state(batch, (), None, None)
