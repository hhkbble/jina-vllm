import asyncio
from dataclasses import dataclass, fields

from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async

from .contract import ARCH, validate_model_contract
from .io_processor import Jina35ScoringIOProcessor
from .ranking import RankingPlan

# vLLM configures handlers on its own logging namespace.
logger = init_logger("vllm.jina_vllm.serving")


@dataclass(kw_only=True)
class Jina35ServeContext(PoolingServeContext):
    plan: RankingPlan | None = None


class Jina35ServingScores(ServingScores):
    def init_io_processor(self, *args, **kwargs):
        return Jina35ScoringIOProcessor(*args, **kwargs)

    async def _init_ctx(self, io_processor, request, raw_request=None):
        base = await super()._init_ctx(io_processor, request, raw_request)
        return Jina35ServeContext(
            **{
                field.name: getattr(base, field.name)
                for field in fields(PoolingServeContext)
            }
        )

    async def _preprocessing(self, io_processor, ctx):
        requests = await make_async(
            io_processor.get_request_factory_online, executor=self._executor
        )(ctx)
        ctx.engine_inputs = await asyncio.gather(
            *(io_processor.render_async(item) for item in requests)
        )
        execution_id = random_uuid()
        ctx.prompt_request_ids = [
            f"{ctx.request_id}-{execution_id}-{index}"
            for index in range(len(ctx.engine_inputs))
        ]

    async def _collect_batch(self, ctx):
        try:
            await super()._collect_batch(ctx)
        except BaseException as error:
            await self._cleanup(ctx)
            # Native collector uses ValueError, which HTTP classifies as 400.
            # An incomplete engine result batch is an internal failure.
            if isinstance(error, ValueError) and str(error) == (
                "Failed to generate results for all prompts"
            ):
                raise RuntimeError(str(error)) from error
            raise

    async def _prepare_generators(self, ctx):
        try:
            await super()._prepare_generators(ctx)
        except BaseException:
            await self._cleanup(ctx)
            raise

    async def _cleanup(self, ctx):
        try:
            if ctx.result_generator is not None:
                await ctx.result_generator.aclose()
        except BaseException:
            logger.exception("Failed to close Jina block iterator")
        try:
            if ctx.prompt_request_ids:
                await self.engine_client.abort(ctx.prompt_request_ids)
                logger.info(
                    "Aborted Jina request %s (%d blocks)",
                    ctx.request_id,
                    len(ctx.prompt_request_ids),
                )
        except BaseException:
            logger.exception("Failed to abort Jina block requests")


async def install_serving(engine_client, state) -> None:
    if engine_client.model_config.architecture != ARCH:
        return
    validate_model_contract(engine_client.vllm_config)
    old = getattr(state, "serving_scores", None)
    if isinstance(old, Jina35ServingScores):
        return
    if not isinstance(old, ServingScores):
        raise RuntimeError("Expected native ServingScores")
    state.serving_scores = Jina35ServingScores(
        engine_client,
        old.models,
        supported_tasks=await engine_client.get_supported_tasks(),
        enable_flash_late_interaction=False,
        request_logger=old.request_logger,
        chat_template_config=old.chat_template_config,
        return_tokens_as_token_ids=old.return_tokens_as_token_ids,
        log_error_stack=old.log_error_stack,
    )
    logger.info("Installed Jina v3.5 native scoring adapter")
