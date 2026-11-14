from copy import deepcopy
from types import SimpleNamespace

import pytest
from transformers import AutoConfig

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


@pytest.fixture
def model_contract(snapshot):
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.embedding_size = 512
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, disable_sliding_window=False),
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
        ("rope_parameters", {"rope_theta": 10000, "rope_type": "default"}),
    ],
)
def test_model_semantic_mismatches(model_contract, field, value):
    setattr(model_contract.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="contract"):
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
