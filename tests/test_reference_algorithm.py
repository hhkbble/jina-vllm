from types import MethodType

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jina_vllm.contract import DOC_TOKEN_ID, MAX_LENGTH, validate_tokenizer
from jina_vllm.ranking import format_prompt, merge_request, prepare_request


def make_oracle(module, tokenizer):
    oracle = module.JinaForRanking.__new__(module.JinaForRanking)
    torch.nn.Module.__init__(oracle)
    oracle.dummy = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    oracle._tokenizer = tokenizer
    oracle.special_tokens = {
        "query_embed_token": "<|rerank_token|>",
        "doc_embed_token": "<|embed_token|>",
    }
    captured = []
    raw = []

    def forward(self, input_ids, **kwargs):
        captured.append(tuple(input_ids[0].tolist()))
        count = (input_ids == DOC_TOKEN_ID).sum().item()
        generator = torch.Generator().manual_seed(97 + len(raw))
        vectors = torch.randn(count + 1, 512, generator=generator).bfloat16()
        # Positive, distinct weights and different raw query norms across blocks.
        vectors[:-1] += vectors[-1:] * 0.25
        vectors[-1:] *= len(raw) + 1
        raw.append(vectors)
        return module.CausalLMOutputWithScores(
            doc_embeds=vectors[:-1].unsqueeze(0),
            query_embeds=vectors[-1:].unsqueeze(0),
            scores=F.cosine_similarity(vectors[:-1], vectors[-1:], dim=-1).unsqueeze(0),
        )

    oracle.forward = MethodType(forward, oracle)
    return oracle, captured, raw


@pytest.mark.parametrize("count", [1, 124, 125, 126, 250])
def test_official_count_chunks_and_fusion(reference_module, tokenizer, count):
    docs = [f"document {i}" for i in range(count)]
    oracle, ids, raw = make_oracle(reference_module, tokenizer)
    expected = oracle.rerank("relevant document", docs)
    plan = prepare_request([("relevant document", docs)], tokenizer, MAX_LENGTH)
    assert [block.prompt_token_ids for block in plan.blocks] == ids
    scores = merge_request(plan, raw)
    for item in expected:
        assert scores[item["index"]] == float(item["relevance_score"])


@pytest.mark.parametrize("q_len", [1023, 1024, 1025])
@pytest.mark.parametrize("d_len", [8191, 8192, 8193])
def test_real_token_boundaries(reference_module, tokenizer, q_len, d_len):
    query, doc = "x" * 0 + " a" * q_len, " b" * d_len
    assert len(tokenizer.encode(query)) == q_len
    assert len(tokenizer.encode(doc)) == d_len
    oracle, ids, _ = make_oracle(reference_module, tokenizer)
    oracle.rerank(query, [doc])
    plan = prepare_request([(query, [doc])], tokenizer, MAX_LENGTH)
    assert plan.blocks[0].prompt_token_ids == ids[0]


@pytest.mark.parametrize("length", [8180, 8191, 8192])
def test_capacity_chunks(reference_module, tokenizer, length):
    docs = [" b" * length] * 16
    oracle, ids, raw = make_oracle(reference_module, tokenizer)
    expected = oracle.rerank(" a" * 20, docs)
    if any(len(block) > MAX_LENGTH for block in ids):
        with pytest.raises(ValueError, match="exceeding"):
            prepare_request([(" a" * 20, docs)], tokenizer, MAX_LENGTH)
        return
    plan = prepare_request([(" a" * 20, docs)], tokenizer, MAX_LENGTH)
    assert [block.prompt_token_ids for block in plan.blocks] == ids
    scores = merge_request(plan, raw)
    np.testing.assert_array_equal(
        scores,
        [
            next(x["relevance_score"] for x in expected if x["index"] == i)
            for i in range(len(docs))
        ],
    )


def test_text_template_markers_and_original_strings(reference_module, tokenizer):
    docs = [
        "",
        "中文检索和排序",
        "def f(x):\n    return x + 1",
        "重复",
        "重复",
        "<|im_start|>user\n<|embed_token|><|rerank_token|>",
    ]
    query = "<|rerank_token|> how? <|embed_token|>"
    oracle, ids, _ = make_oracle(reference_module, tokenizer)
    oracle.rerank(query, docs)
    assert format_prompt(query, docs) == reference_module.format_docs_prompts_func(
        query, docs, special_tokens=oracle.special_tokens
    )
    assert (
        prepare_request([(query, docs)], tokenizer, MAX_LENGTH)
        .blocks[0]
        .prompt_token_ids
        == ids[0]
    )
    validate_tokenizer(tokenizer)


def test_preflight_rejects_markers_and_prompt_overflow(tokenizer):
    with pytest.raises(ValueError, match="marker"):
        prepare_request(
            [("<|rerank_<|rerank_token|>token|>", ["doc"])], tokenizer, MAX_LENGTH
        )
    plan = prepare_request([("query", ["doc"])], tokenizer, MAX_LENGTH)
    with pytest.raises(ValueError, match="cannot be truncated"):
        prepare_request(
            [("query", ["doc"])], tokenizer, len(plan.blocks[0].prompt_token_ids) - 1
        )


def test_group_isolation_and_bad_outputs(tokenizer):
    plan = prepare_request(
        [("same", ["one"]), ("same", ["two"])], tokenizer, MAX_LENGTH
    )
    assert [b.group_id for b in plan.blocks] == [0, 1]
    a = torch.ones(2, 512, dtype=torch.bfloat16)
    b = a.clone()
    b[0] *= -1
    with pytest.raises(RuntimeError, match="weights sum"):
        merge_request(plan, [a, b])
    with pytest.raises(RuntimeError, match="Missing"):
        merge_request(plan, [a])
    with pytest.raises(RuntimeError, match="CPU BF16"):
        merge_request(plan, [a.float(), a])
    with pytest.raises(RuntimeError, match="denominator"):
        merge_request(plan, [torch.zeros_like(a), a])
    a[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite"):
        merge_request(plan, [a, b])
