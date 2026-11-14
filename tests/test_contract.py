from copy import deepcopy
from types import SimpleNamespace

import pytest
from transformers import AutoConfig
from vllm.config import CompilationMode
from vllm.config.compilation import CompilationConfig
from vllm.config.vllm import OptimizationLevel

from jina_vllm.reranker_v35.config import RerankerV35Config, reranker_block_size
from jina_vllm.reranker_v35.contract import validate_model_contract


def test_block_size_configuration():
    assert reranker_block_size({"other_extension": {"key": 1}}) == 125
    assert reranker_block_size({"jina_vllm": {"reranker_v35": {"block_size": 1}}}) == 1


@pytest.mark.parametrize("value", [0, -1, True, 125.0, "125", None])
def test_block_size_requires_positive_integer(value):
    config = {"jina_vllm": {"reranker_v35": {"block_size": value}}}
    with pytest.raises(ValueError, match="positive integer"):
        reranker_block_size(config)


@pytest.fixture
def cuda_compile_config(monkeypatch):
    monkeypatch.setattr(
        "jina_vllm.reranker_v35.config.current_platform.is_cuda", lambda: True
    )
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(enforce_eager=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=349_526),
        compilation_config=CompilationConfig(
            backend="inductor",
            pass_config={"enable_qk_norm_rope_fusion": False},
        ),
        optimization_level=OptimizationLevel.O2,
    )


@pytest.mark.parametrize(
    ("batch_tokens", "custom_ops", "expected"),
    [
        (349_525, ["none", "-silu_and_mul"], ["none", "-silu_and_mul"]),
        (349_526, ["none"], ["none", "+silu_and_mul"]),
        (524_288, [], ["+silu_and_mul"]),
        (524_288, ["all"], ["all"]),
        (524_288, ["none", "+silu_and_mul"], ["none", "+silu_and_mul"]),
    ],
)
def test_swiglu_selection_by_batch_budget(
    cuda_compile_config, batch_tokens, custom_ops, expected
):
    cuda_compile_config.scheduler_config.max_num_batched_tokens = batch_tokens
    compilation = cuda_compile_config.compilation_config
    compilation.custom_ops = custom_ops.copy()
    RerankerV35Config.verify_and_update_config(cuda_compile_config)
    assert compilation.custom_ops == expected


def test_large_inductor_batch_requires_native_swiglu(cuda_compile_config):
    cuda_compile_config.compilation_config.custom_ops = ["all", "-silu_and_mul"]
    with pytest.raises(ValueError, match=r"349,525.*silu_and_mul"):
        RerankerV35Config.verify_and_update_config(cuda_compile_config)


@pytest.mark.parametrize(
    ("mode", "backend", "enforce_eager", "optimization_level"),
    [
        (CompilationMode.NONE, "inductor", False, OptimizationLevel.O2),
        (CompilationMode.VLLM_COMPILE, "eager", False, OptimizationLevel.O2),
        (None, "inductor", True, OptimizationLevel.O2),
        (None, "inductor", False, OptimizationLevel.O0),
    ],
)
def test_swiglu_opt_out_without_inductor(
    cuda_compile_config, mode, backend, enforce_eager, optimization_level
):
    cuda_compile_config.model_config.enforce_eager = enforce_eager
    cuda_compile_config.optimization_level = optimization_level
    compilation = cuda_compile_config.compilation_config
    compilation.mode = mode
    compilation.backend = backend
    compilation.custom_ops = ["-silu_and_mul"]
    RerankerV35Config.verify_and_update_config(cuda_compile_config)
    assert compilation.custom_ops == ["-silu_and_mul"]


@pytest.fixture
def model_contract(snapshot):
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.embedding_size = 512
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            disable_sliding_window=False,
            max_model_len=131072,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        cache_config=SimpleNamespace(sliding_window=None),
    )


def test_normalized_rope_and_export_metadata(model_contract):
    config = model_contract.model_config.hf_config
    assert config.rope_parameters == {"rope_theta": 1000000, "rope_type": "default"}
    config._name_or_path = "portable mirror"
    config.some_export_metadata = "unused"
    validate_model_contract(model_contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_dim", 64),
        ("embedding_size", 1024),
        ("layer_types", ["full_attention"] * 28),
        ("is_causal", False),
        ("max_window_layers", 16),
    ],
)
def test_model_semantic_mismatches(model_contract, field, value):
    setattr(model_contract.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="contract"):
        validate_model_contract(model_contract)


@pytest.mark.parametrize("engine_limit", [131072, 131073])
def test_yarn_reranker_contract(model_contract, engine_limit):
    model = model_contract.model_config
    model.max_model_len = engine_limit
    if engine_limit > 131072:
        model.hf_config.max_position_embeddings = 262144
    model.hf_config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 1000000,
        "original_max_position_embeddings": 131072,
        "factor": 2.0,
    }
    validate_model_contract(model_contract)


@pytest.mark.parametrize(
    "override, error",
    [
        ({"rope_theta": 3500000}, "rope_theta=1000000"),
        ({"original_max_position_embeddings": 32768}, "131072"),
        ({"factor": float("inf")}, "finite"),
        ({"factor": 1.0}, "greater than 1"),
        ({"rope_type": "linear"}, "default or yarn"),
        ({"beta_fast": 16}, "Jina YaRN requires"),
    ],
)
def test_yarn_rejects_invalid_parameters(model_contract, override, error):
    model_contract.model_config.hf_config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 1000000,
        "original_max_position_embeddings": 131072,
        "factor": 2.0,
        **override,
    }
    with pytest.raises(ValueError, match=error):
        validate_model_contract(model_contract)


@pytest.mark.parametrize("kind", ["disabled", "uniform", "tp", "pp"])
def test_invalid_engine_windows_and_parallelism(model_contract, kind):
    config = deepcopy(model_contract)
    if kind == "disabled":
        config.model_config.disable_sliding_window = True
    if kind == "uniform":
        config.cache_config.sliding_window = 1024
    if kind == "tp":
        config.parallel_config.tensor_parallel_size = 2
    if kind == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="contract"):
        validate_model_contract(config)
