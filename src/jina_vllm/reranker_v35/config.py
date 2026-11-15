from vllm.config import VllmConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from jina_vllm.idle_trim import validate_idle_trim_config

from .contract import DEFAULT_BLOCK_SIZE, validate_block_size


def reranker_block_size(additional_config) -> int:
    options = additional_config if isinstance(additional_config, dict) else {}
    jina = options.get("jina_vllm", {})
    if not isinstance(jina, dict):
        raise ValueError("additional_config.jina_vllm must be an object")
    reranker = jina.get("reranker_v35", {})
    if not isinstance(reranker, dict) or reranker.keys() - {"block_size"}:
        raise ValueError("additional_config.jina_vllm.reranker_v35 accepts block_size")
    block_size = reranker.get("block_size", DEFAULT_BLOCK_SIZE)
    validate_block_size(block_size)
    return block_size


class RerankerV35Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        reranker_block_size(vllm_config.additional_config)
        validate_idle_trim_config(vllm_config)
        passes = vllm_config.compilation_config.pass_config
        if passes.fuse_act_quant is None:
            passes.fuse_act_quant = False
        if passes.fuse_norm_quant is None:
            passes.fuse_norm_quant = False
