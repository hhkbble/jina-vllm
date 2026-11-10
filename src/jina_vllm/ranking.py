"""Official v3.5 request planning and raw-vector fusion, independent of vLLM.

Template and algorithm adapted from jinaai/jina-reranker-v3.5 modeling.py,
revision e8a93f33f0b22108f8c2364f8484ce3422552fbc.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .contract import (
    DOC_MARKER,
    DOC_TOKEN_ID,
    MAX_LENGTH,
    QUERY_MARKER,
    QUERY_TOKEN_ID,
)


@dataclass(frozen=True)
class BlockPlan:
    group_id: int
    output_indices: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class RankingPlan:
    num_outputs: int
    blocks: tuple[BlockPlan, ...]


def format_prompt(query: str, documents: Sequence[str]) -> str:
    def sanitize(text: str) -> str:
        return text.replace(QUERY_MARKER, "").replace(DOC_MARKER, "")

    query = sanitize(query)
    documents = [sanitize(doc) for doc in documents]
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the "
        "passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how "
        "well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well "
        "each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction "
        "when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    prompt = (
        f"I will provide you with {len(documents)} passages, each indicated "
        f"by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    prompt += (
        "\n".join(
            f'<passage id="{i}">\n{doc}{DOC_MARKER}\n</passage>'
            for i, doc in enumerate(documents)
        )
        + "\n"
    )
    prompt += f"<query>\n{query}{QUERY_MARKER}\n</query>"
    return (
        prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _truncate(text: str, limit: int, tokenizer: Any) -> tuple[str, int]:
    ids = tokenizer(text, truncation=True, max_length=limit)["input_ids"]
    return (tokenizer.decode(ids) if len(ids) >= limit else text), len(ids)


def prepare_request(
    groups: Sequence[tuple[str, Sequence[str]]],
    tokenizer: Any,
    engine_max_length: int,
) -> RankingPlan:
    blocks = []
    offset = 0
    if not groups:
        raise ValueError("At least one query/document group is required")

    for group_id, (query, documents) in enumerate(groups):
        if not documents:
            raise ValueError("Each query requires at least one document")
        prepared = [_truncate(doc, 8192, tokenizer) for doc in documents]
        query, query_length = _truncate(query, 1024, tokenizer)
        capacity = MAX_LENGTH - 2 * query_length
        current: list[str] = []
        indices: list[int] = []

        def emit(query, current, indices, group_id) -> None:
            ids = tokenizer.encode(format_prompt(query, current))
            if len(ids) > min(MAX_LENGTH, engine_max_length):
                raise ValueError(
                    f"Official Jina block has {len(ids)} tokens, exceeding "
                    f"the engine limit {min(MAX_LENGTH, engine_max_length)}; "
                    "final prompts cannot be truncated or repartitioned"
                )
            if (
                ids.count(DOC_TOKEN_ID) != len(current)
                or ids.count(QUERY_TOKEN_ID) != 1
            ):
                raise ValueError("Jina prompt marker counts do not match the block")
            blocks.append(BlockPlan(group_id, tuple(indices), tuple(ids)))

        for index, (document, length) in enumerate(prepared):
            current.append(document)
            indices.append(offset + index)
            capacity -= length
            if len(current) >= 125 or capacity <= 8192:
                emit(query, current, indices, group_id)
                current = []
                indices = []
                capacity = MAX_LENGTH - 2 * query_length
        if current:
            emit(query, current, indices, group_id)
        offset += len(documents)

    return RankingPlan(offset, tuple(blocks))


@torch.inference_mode()
def merge_request(
    plan: RankingPlan, raw_block_tensors: Sequence[torch.Tensor]
) -> list[float]:
    if len(raw_block_tensors) != len(plan.blocks):
        raise RuntimeError("Missing or excess Jina block outputs")
    groups: dict[int, list[tuple[BlockPlan, torch.Tensor]]] = {}
    for block, vectors in zip(plan.blocks, raw_block_tensors):
        if (
            vectors.device.type != "cpu"
            or vectors.dtype != torch.bfloat16
            or vectors.shape != (len(block.output_indices) + 1, 512)
        ):
            raise RuntimeError(
                "Jina block output must be a CPU BF16 tensor with shape "
                f"{(len(block.output_indices) + 1, 512)}, received "
                f"{vectors.device}/{vectors.dtype}/{tuple(vectors.shape)}"
            )
        groups.setdefault(block.group_id, []).append((block, vectors))

    result = np.empty(plan.num_outputs, dtype=np.float32)
    for group in groups.values():
        queries, documents, weights, indices = [], [], [], []
        for block, vectors in group:
            local_scores = (
                F.cosine_similarity(vectors[:-1], vectors[-1:], dim=-1).float().numpy()
            )
            queries.append(vectors[-1:].float().numpy())
            documents.append(vectors[:-1].float().numpy())
            weights.append(((np.float32(1) + local_scores) / np.float32(2)).max())
            indices.extend(block.output_indices)
        weight_array = np.asarray(weights, dtype=np.float32)
        if weight_array.sum() == 0:
            raise RuntimeError("Jina block weights sum to zero")
        query = np.average(np.stack(queries), axis=0, weights=weight_array)
        document_matrix = np.concatenate(documents)
        try:
            with np.errstate(divide="raise", invalid="raise"):
                scores = np.dot(query, document_matrix.T) / (
                    np.linalg.norm(query) * np.linalg.norm(document_matrix, axis=1)
                )
        except FloatingPointError as error:
            raise RuntimeError("Invalid Jina cosine denominator or vectors") from error
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite Jina relevance scores")
        result[indices] = scores.reshape(-1)
    return result.tolist()
