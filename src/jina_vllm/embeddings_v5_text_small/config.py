from vllm.config import VllmConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from jina_vllm.config import default_qk_norm_rope_fusion


class TextSmallConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        default_qk_norm_rope_fusion(vllm_config)
