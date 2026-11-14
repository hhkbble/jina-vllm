from vllm.config import CompilationMode, VllmConfig
from vllm.config.vllm import OptimizationLevel
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.platforms import current_platform

from .contract import DEFAULT_BLOCK_SIZE, INTERMEDIATE_SIZE, validate_block_size

# The packed gate/up tensor has two intermediate-width halves per token.
_MAX_INT32_GATE_UP_ROWS = (1 << 31) // (2 * INTERMEDIATE_SIZE)


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
        compilation = vllm_config.compilation_config
        ops = compilation.custom_ops
        mode = compilation.mode
        if mode is None:
            # vLLM resolves the default mode after running model config hooks.
            mode = (
                CompilationMode.NONE
                if vllm_config.optimization_level == OptimizationLevel.O0
                else CompilationMode.VLLM_COMPILE
            )
        if (
            current_platform.is_cuda()
            and compilation.backend == "inductor"
            and mode != CompilationMode.NONE
            and not vllm_config.model_config.enforce_eager
            and vllm_config.scheduler_config.max_num_batched_tokens
            > _MAX_INT32_GATE_UP_ROWS
        ):
            # Inductor can fuse SwiGLU with FP8 quantization using int32 offsets.
            # The native CUDA op uses int64 offsets, including during warmup.
            if "-silu_and_mul" in ops:
                raise ValueError(
                    f"Jina v3.5 CUDA Inductor batches above {_MAX_INT32_GATE_UP_ROWS:,}"
                    " tokens require the native silu_and_mul op. Remove "
                    "'-silu_and_mul' from compilation_config.custom_ops."
                )
            if "all" not in ops and "+silu_and_mul" not in ops:
                ops.append("+silu_and_mul")

        passes = compilation.pass_config
        if passes.fuse_act_quant is None:
            passes.fuse_act_quant = False
        if passes.fuse_norm_quant is None:
            passes.fuse_norm_quant = False
        if passes.enable_qk_norm_rope_fusion is None:
            # Mirror the custom RoPE selection used by the pinned vLLM
            # CompilationConfig when the QK pass is explicitly enabled.
            passes.enable_qk_norm_rope_fusion = (
                (current_platform.is_cuda_alike() or current_platform.is_xpu())
                and "-rotary_embedding" not in ops
                and not vllm_config.model_config.enforce_eager
                and compilation.mode in (None, CompilationMode.VLLM_COMPILE)
            )
            if passes.enable_qk_norm_rope_fusion and "+rotary_embedding" not in ops:
                ops.append("+rotary_embedding")
