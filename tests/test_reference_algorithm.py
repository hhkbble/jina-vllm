from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from jina_vllm.reranker_v35.contract import DOC_TOKEN_ID, MAX_LENGTH, validate_tokenizer
from jina_vllm.reranker_v35.ranking import format_prompt, merge_request, prepare_request


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


def test_configured_block_fits_300_documents(tokenizer):
    docs = [f"document {i}" for i in range(300)]
    plan = prepare_request([("query", docs)], tokenizer, MAX_LENGTH, block_size=300)
    (block,) = plan.blocks
    assert block.output_indices == tuple(range(300))
    expected = tokenizer.encode(format_prompt("query", docs))
    assert block.prompt_token_ids == tuple(expected)


def test_exact_capacity_and_one_token_split(reference_module, tokenizer):
    query = "query"
    docs = ["first document", " b" * 35]
    oracle, ids, _ = make_oracle(reference_module, tokenizer)
    oracle.rerank(query, docs)
    (whole,) = ids
    limit = len(whole)
    plan = prepare_request([(query, docs)], tokenizer, limit)
    assert plan.blocks[0].prompt_token_ids == whole

    longer = [docs[0], docs[1] + " b"]
    assert len(tokenizer.encode(format_prompt(query, longer))) == limit + 1
    plan = prepare_request([(query, longer)], tokenizer, limit)
    assert [b.output_indices for b in plan.blocks] == [(0,), (1,)]
    assert [b.prompt_token_ids for b in plan.blocks] == [
        tuple(tokenizer.encode(format_prompt(query, [doc]))) for doc in longer
    ]


def test_full_context_uses_actual_template_budget(tokenizer):
    query, docs = " a" * 20, [" b" * 8192] * 16
    plan = prepare_request([(query, docs)], tokenizer, MAX_LENGTH)
    assert [len(b.output_indices) for b in plan.blocks] == [15, 1]
    assert all(len(b.prompt_token_ids) <= MAX_LENGTH for b in plan.blocks)
    assert len(tokenizer.encode(format_prompt(query, docs))) > MAX_LENGTH


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
    query, doc = "query", " b" * 100
    length = len(tokenizer.encode(format_prompt(query, [doc])))
    with pytest.raises(
        ValueError, match=f"index 1 needs {length} prompt tokens.*{length - 1}"
    ):
        prepare_request([(query, ["short", doc])], tokenizer, length - 1)


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
