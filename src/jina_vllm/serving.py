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


async def _close_iterators(iterators):
    for iterator in iterators:
        try:
            await iterator.aclose()
        except BaseException:
            logger.exception("Failed to close Jina block producer")


async def _merge_blocks(iterators):
    # Adapted from vLLM merge_async_iterators (Apache-2.0, see NOTICE).
    # Join cancelled anext tasks before closing their running generators.
    pending = {}
    finished = set()
    try:
        for index, iterator in enumerate(iterators):
            pending[asyncio.create_task(anext(iterator))] = index
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = pending.pop(task)
                try:
                    result = task.result()
                except StopAsyncIteration:
                    continue
                if index in finished:
                    raise RuntimeError("Duplicate completed Jina block output")
                if result.finished:
                    finished.add(index)
                pending[asyncio.create_task(anext(iterators[index]))] = index
                yield index, result
    finally:
        tasks = list(pending)
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, (asyncio.CancelledError, StopAsyncIteration)
            ):
                logger.error(
                    "Jina block producer failed while draining",
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
        await _close_iterators(iterators)


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
        # Keep the pinned native forwarding contract, replacing only its merger.
        iterators = []
        try:
            assert ctx.engine_inputs is not None
            assert ctx.prompt_request_ids is not None
            headers = (
                None
                if ctx.raw_request is None
                else await self._get_trace_headers(ctx.raw_request.headers)
            )
            ctx.pooling_params.verify(self.model_config)
            for request_id, engine_input in zip(
                ctx.prompt_request_ids, ctx.engine_inputs
            ):
                self._log_inputs(
                    request_id,
                    engine_input["prompts"],
                    params=engine_input["params"],
                    lora_request=ctx.lora_request,
                )
                iterators.append(
                    self.engine_client.encode(
                        request_id=request_id,
                        prompt=engine_input["prompts"],
                        pooling_params=engine_input["params"],
                        lora_request=engine_input["lora_requests"],
                        priority=engine_input["priorities"],
                        trace_headers=headers,
                    )
                )
            ctx.result_generator = _merge_blocks(iterators)
        except BaseException:
            await _close_iterators(iterators)
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
