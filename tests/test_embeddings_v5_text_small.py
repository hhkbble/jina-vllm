from types import SimpleNamespace

import pytest
from vllm.config import CompilationConfig, CompilationMode

from jina_vllm.embeddings_v5_text_small.config import TextSmallConfig
from jina_vllm.embeddings_v5_text_small.model import validate_model_contract


@pytest.mark.parametrize(
    ("compile_options", "eager", "cuda_alike", "expected"),
    [
        pytest.param({}, False, True, True, id="default"),
        pytest.param(
            {"mode": CompilationMode.VLLM_COMPILE, "custom_ops": ["+rotary_embedding"]},
            False,
            True,
            True,
            id="existing-custom-op",
        ),
        pytest.param(
            {"pass_config": {"enable_qk_norm_rope_fusion": False}},
            False,
            True,
            False,
            id="explicit-opt-out",
        ),
        pytest.param(
            {"pass_config": {"enable_qk_norm_rope_fusion": True}},
            False,
            True,
            True,
            id="native-opt-in",
        ),
        pytest.param(
            {"custom_ops": ["-rotary_embedding"]},
            False,
            True,
            False,
            id="custom-op-opt-out",
        ),
        pytest.param({}, True, True, False, id="eager"),
        pytest.param(
            {"mode": CompilationMode.NONE}, False, True, False, id="no-compilation"
        ),
        pytest.param({}, False, False, False, id="unsupported-platform"),
    ],
)
def test_text_small_qk_fusion_default(
    monkeypatch, compile_options, eager, cuda_alike, expected
):
    monkeypatch.setattr(
        "jina_vllm.config.current_platform.is_cuda_alike", lambda: cuda_alike
    )
    monkeypatch.setattr("jina_vllm.config.current_platform.is_xpu", lambda: False)
    compilation = CompilationConfig(**compile_options)
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=compilation,
    )
    TextSmallConfig.verify_and_update_config(config)
    assert compilation.pass_config.enable_qk_norm_rope_fusion is expected
    assert compilation.custom_ops.count("+rotary_embedding") == int(expected)
    if "-rotary_embedding" in compile_options.get("custom_ops", []):
        assert "-rotary_embedding" in compilation.custom_ops
    assert compilation.pass_config.fuse_act_quant is None
    assert compilation.pass_config.fuse_norm_quant is None


@pytest.fixture
def text_small_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=32768,
            hf_config=SimpleNamespace(
                model_type="jina_embeddings_v5",
                hidden_size=1024,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=32768,
                rope_parameters={"rope_type": "default", "rope_theta": 3500000},
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )


def test_text_small_accepts_decoder_checkpoint(text_small_config):
    validate_model_contract(text_small_config)


@pytest.mark.parametrize("positions", [32768, 65536, 65537])
def test_text_small_yarn_position_budget(text_small_config, positions):
    model = text_small_config.model_config
    model.hf_config.max_position_embeddings = positions
    model.max_model_len = 32769
    model.hf_config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 3500000,
        "original_max_position_embeddings": 32768,
        "factor": 2.0,
    }
    if positions == 65536:
        validate_model_contract(text_small_config)
    else:
        with pytest.raises(ValueError, match="max_position_embeddings"):
            validate_model_contract(text_small_config)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("num_key_value_heads", 4, "num_key_value_heads"),
        ("head_dim", 64, "head_dim"),
        ("is_decoder", False, "full causal"),
        ("is_causal", False, "full causal"),
        ("sliding_window", 1024, "full causal"),
    ],
)
def test_text_small_rejects_incompatible_checkpoint(
    text_small_config, field, value, error
):
    setattr(text_small_config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match=error):
        validate_model_contract(text_small_config)


def test_text_small_rejects_tensor_parallel(text_small_config):
    text_small_config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="parallel size 1"):
        validate_model_contract(text_small_config)
