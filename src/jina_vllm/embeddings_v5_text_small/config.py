from vllm.config import VllmConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from jina_vllm.idle_trim import validate_idle_trim_config


class TextSmallConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        validate_idle_trim_config(vllm_config)
