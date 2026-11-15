from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from vllm.config import CompilationMode
from vllm.config.compilation import CompilationConfig
from vllm.engine.arg_utils import EngineArgs

from jina_vllm.config import configure_large_batch_swiglu
from jina_vllm.plugin import PLUGIN_NAME, register
from jina_vllm.reranker_v35.config import reranker_block_size
from jina_vllm.reranker_v35.contract import validate_model_contract


def test_block_size_configuration():
    assert reranker_block_size({"other_extension": {"key": 1}}) == 125
    assert reranker_block_size({"jina_vllm": {"reranker_v35": {"block_size": 1}}}) == 1


@pytest.mark.parametrize("value", [0, -1, True, 125.0, "125", None])
def test_block_size_requires_positive_integer(value):
    config = {"jina_vllm": {"reranker_v35": {"block_size": value}}}
    with pytest.raises(ValueError, match="positive integer"):
        reranker_block_size(config)


@pytest.mark.parametrize(
    ("engine_options", "compile_options", "disable_compile", "mode", "fusion"),
    [
        ({}, {}, False, CompilationMode.VLLM_COMPILE, True),
        ({"optimization_level": 0}, {}, False, CompilationMode.NONE, False),
        ({"enforce_eager": True}, {}, False, CompilationMode.NONE, False),
        ({}, {"backend": "eager"}, False, CompilationMode.VLLM_COMPILE, False),
        ({}, {}, True, CompilationMode.NONE, False),
        (
            {},
            {"pass_config": {"enable_qk_norm_rope_fusion": False}},
            False,
            CompilationMode.VLLM_COMPILE,
            False,
        ),
        (
            {"optimization_level": 0},
            {"pass_config": {"enable_qk_norm_rope_fusion": True}},
            False,
            CompilationMode.NONE,
            True,
        ),
        (
            {},
            {"custom_ops": ["none", "-rotary_embedding"]},
            False,
            CompilationMode.VLLM_COMPILE,
            False,
        ),
    ],
)
def test_qk_fusion_follows_resolved_compilation(
    snapshot,
    monkeypatch,
    engine_options,
    compile_options,
    disable_compile,
    mode,
    fusion,
):
    monkeypatch.setenv("VLLM_PLUGINS", PLUGIN_NAME)
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", str(int(disable_compile)))
    monkeypatch.setattr("jina_vllm.config.current_platform.is_cuda_alike", lambda: True)
    register()
    config = EngineArgs(
        model=str(snapshot),
        model_impl="vllm",
        runner="pooling",
        dtype="bfloat16",
        max_model_len=8192,
        hf_overrides={"architectures": ["JinaRerankerV35"], "embedding_size": 512},
        pooler_config={"task": "token_embed"},
        compilation_config={"custom_ops": ["none"], **compile_options},
        **engine_options,
    ).create_engine_config()
    compilation = config.compilation_config
    assert compilation.mode == mode
    assert compilation.pass_config.enable_qk_norm_rope_fusion is fusion
    assert compilation.custom_ops.count("+rotary_embedding") == int(fusion)


@pytest.fixture
def cuda_compile_config(monkeypatch):
    monkeypatch.setattr("jina_vllm.config.current_platform.is_cuda", lambda: True)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="JinaEmbeddingsV5TextSmall",
            hf_config=SimpleNamespace(intermediate_size=3072),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=349_526),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
        ),
    )


@pytest.mark.parametrize(
    ("intermediate_size", "batch_tokens", "custom_ops", "expected"),
    [
        (3072, 349_525, ["none", "-silu_and_mul"], ["none", "-silu_and_mul"]),
        (3072, 349_526, ["none"], ["none", "+silu_and_mul"]),
        (3072, 524_288, ["all"], ["all"]),
        (3072, 524_288, ["none", "+silu_and_mul"], ["none", "+silu_and_mul"]),
        (4096, 262_144, ["none"], ["none"]),
        (4096, 262_145, ["none"], ["none", "+silu_and_mul"]),
    ],
)
def test_swiglu_selection_by_batch_budget(
    cuda_compile_config, intermediate_size, batch_tokens, custom_ops, expected
):
    cuda_compile_config.model_config.hf_config.intermediate_size = intermediate_size
    cuda_compile_config.scheduler_config.max_num_batched_tokens = batch_tokens
    compilation = cuda_compile_config.compilation_config
    compilation.custom_ops = custom_ops.copy()
    configure_large_batch_swiglu(cuda_compile_config)
    assert compilation.custom_ops == expected


def test_large_inductor_batch_requires_native_swiglu(cuda_compile_config):
    cuda_compile_config.compilation_config.custom_ops = ["all", "-silu_and_mul"]
    with pytest.raises(ValueError, match=r"349,525.*silu_and_mul"):
        configure_large_batch_swiglu(cuda_compile_config)


@pytest.mark.parametrize(
    ("mode", "backend"),
    [
        (CompilationMode.NONE, "inductor"),
        (CompilationMode.VLLM_COMPILE, "eager"),
    ],
)
def test_swiglu_opt_out_without_inductor(cuda_compile_config, mode, backend):
    compilation = cuda_compile_config.compilation_config
    compilation.mode = mode
    compilation.backend = backend
    compilation.custom_ops = ["-silu_and_mul"]
    configure_large_batch_swiglu(cuda_compile_config)
    assert compilation.custom_ops == ["-silu_and_mul"]


@pytest.fixture
def model_contract(snapshot):
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.embedding_size = 512
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
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


def test_reranker_rejects_fp16_marker_vectors(model_contract):
    model_contract.model_config.dtype = torch.float16
    with pytest.raises(ValueError, match="dtype must be bfloat16"):
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
