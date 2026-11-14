from typing import TYPE_CHECKING

import torch
from vllm.entrypoints.pooling.scoring.io_processor import ScoringIOProcessor
from vllm.entrypoints.pooling.typing import EncodeCMPLRenderParams
from vllm.exceptions import VLLMValidationError
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.renderers import TokenizeParams

from .config import reranker_block_size
from .contract import validate_tokenizer
from .ranking import merge_request, prepare_request

if TYPE_CHECKING:
    from .serving import RerankerV35ServeContext


class RerankerV35ScoringIOProcessor(ScoringIOProcessor):
    name = "jina-v35-scoring"
    pooling_task = "token_embed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_tokenizer(self.tokenizer)
        self.block_size = reranker_block_size(self.vllm_config.additional_config)

    def get_request_factory_online(self, ctx: "RerankerV35ServeContext"):
        request = ctx.request
        self._validate_options(request)
        if ctx.lora_request is not None:
            raise VLLMValidationError("Jina v3.5 does not support dynamic LoRA")
        data = self.valid_inputs_online(request)
        groups = (
            [(data.data_1[0], data.data_2)]
            if len(data.data_1) == 1
            else [(query, [doc]) for query, doc in zip(data.data_1, data.data_2)]
        )
        try:
            ctx.plan = prepare_request(
                groups,
                self.tokenizer,
                self.model_config.max_model_len,
                block_size=self.block_size,
            )
        except ValueError as error:
            raise VLLMValidationError(str(error)) from error
        count = len(ctx.plan.blocks)
        params = self._params_to_seq(ctx.pooling_params, count)
        priorities = self._priority_to_seq(ctx.priorities, count)
        tok_params = TokenizeParams(
            max_total_tokens=self.model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=None,
            pad_prompt_tokens=None,
            add_special_tokens=False,
        )
        return [
            EncodeCMPLRenderParams(
                prompts={"prompt_token_ids": list(block.prompt_token_ids)},
                tok_params=tok_params,
                prompt_extras=ctx.prompt_extras,
                skip_mm_cache=False,
                params=params[index],
                lora_requests=None,
                priorities=priorities[index],
            )
            for index, block in enumerate(ctx.plan.blocks)
        ]

    @staticmethod
    def _validate_options(request) -> None:
        supported = {
            "max_tokens_per_query": (0, 1024),
            "max_tokens_per_doc": (0, 8192),
            "instruction": (None, ""),
            "truncate_prompt_tokens": (None,),
            "padding": (None, "do_not_pad"),
            "use_activation": (None, False),
        }
        for field, allowed in supported.items():
            if getattr(request, field, None) not in allowed:
                raise VLLMValidationError(
                    f"Jina v3.5 supports {field} only in {allowed!r}"
                )
        kwargs = request.chat_template_kwargs or {}
        if set(kwargs) - {"instruction"} or kwargs.get("instruction") not in (None, ""):
            raise VLLMValidationError(
                "Jina v3.5 uses a fixed template without instruction"
            )

    def post_process_online(self, ctx: "RerankerV35ServeContext") -> None:
        plan = ctx.plan
        if plan is None:
            raise RuntimeError("Jina request plan is missing")
        outputs = ctx.final_res_batch
        if len(outputs) != len(plan.blocks):
            raise RuntimeError("Missing Jina block outputs")
        for index, output in enumerate(outputs):
            if (
                output.request_id != ctx.prompt_request_ids[index]
                or tuple(output.prompt_token_ids) != plan.blocks[index].prompt_token_ids
                or not output.finished
            ):
                raise RuntimeError("Jina block output identity or completion mismatch")
        scores = merge_request(plan, [output.outputs.data for output in outputs])
        results = [None] * plan.num_outputs
        for block, source in zip(plan.blocks, outputs):
            for index in block.output_indices:
                results[index] = PoolingRequestOutput(
                    request_id=source.request_id,
                    outputs=PoolingOutput(
                        data=torch.tensor(scores[index], dtype=torch.float32)
                    ),
                    prompt_token_ids=source.prompt_token_ids,
                    num_cached_tokens=source.num_cached_tokens,
                    finished=True,
                )
        ctx.final_res_batch = results
