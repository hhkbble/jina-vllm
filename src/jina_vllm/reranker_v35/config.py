from vllm.config import CompilationMode, VllmConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.platforms import current_platform


class RerankerV35Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        compilation = vllm_config.compilation_config
        passes = compilation.pass_config
        if passes.fuse_act_quant is None:
            passes.fuse_act_quant = False
        if passes.fuse_norm_quant is None:
            passes.fuse_norm_quant = False
        if passes.enable_qk_norm_rope_fusion is None:
            ops = compilation.custom_ops
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
