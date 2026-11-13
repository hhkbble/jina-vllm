from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import AttentionConfig, CacheConfig, SchedulerConfig
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from jina_vllm.prefill import (
    PrefillModelState,
    paged_kv_layer_reason,
    paged_kv_reason,
)


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
