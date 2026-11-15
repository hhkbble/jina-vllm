from .embeddings_v5_text_small import ARCH as EMBEDDINGS_V5_TEXT_SMALL_ARCH
from .reranker_v35.contract import ARCH as RERANKER_V35_ARCH

PLUGIN_NAME = "jina_vllm"


def register() -> None:
    from vllm import envs

    allowed = envs.VLLM_PLUGINS
    if allowed is None or PLUGIN_NAME not in allowed:
        return

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import register_prefill_modes
    from .embeddings_v5_text_small.config import TextSmallConfig
    from .idle_trim import install_idle_trim
    from .reranker_v35.config import RerankerV35Config

    ModelRegistry.register_model(
        RERANKER_V35_ARCH, "jina_vllm.reranker_v35.model:JinaRerankerV35"
    )
    ModelRegistry.register_model(
        EMBEDDINGS_V5_TEXT_SMALL_ARCH,
        "jina_vllm.embeddings_v5_text_small.model:JinaEmbeddingsV5TextSmall",
    )
    MODELS_CONFIG_MAP[RERANKER_V35_ARCH] = RerankerV35Config
    MODELS_CONFIG_MAP[EMBEDDINGS_V5_TEXT_SMALL_ARCH] = TextSmallConfig
    register_prefill_modes()
    install_idle_trim()


class JinaEndpointPlugin:
    name = PLUGIN_NAME
    required_tasks = ("token_embed",)

    def attach_router(self, app) -> None:
        pass

    async def init_state(self, engine_client, state, args) -> None:
        if (
            engine_client is None
            or engine_client.model_config.architecture != RERANKER_V35_ARCH
        ):
            return
        from .reranker_v35.serving import install_serving

        await install_serving(engine_client, state)
