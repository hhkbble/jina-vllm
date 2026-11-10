from .contract import ARCH, PLUGIN_NAME


def register() -> None:
    from vllm import envs
    from vllm.model_executor.models import ModelRegistry

    allowed = envs.VLLM_PLUGINS
    if allowed is not None and PLUGIN_NAME in allowed:
        ModelRegistry.register_model(ARCH, "jina_vllm.model:JinaRerankerV35")


class Jina35EndpointPlugin:
    name = PLUGIN_NAME
    required_tasks = ("token_embed",)

    def attach_router(self, app) -> None:
        pass

    async def init_state(self, engine_client, state, args) -> None:
        if engine_client is None or engine_client.model_config.architecture != ARCH:
            return
        from .serving import install_serving

        await install_serving(engine_client, state)
