# jina-reranker-v35-vllm

Out-of-tree support for `jinaai/jina-reranker-v3.5` in vLLM 0.33.0,
using native Qwen3 computation, per-layer attention windows, and the official
listwise truncation, chunking and query-vector fusion algorithm.

**Status:** BF16 passes numerical checks against the fixed official model on
RTX 5090. FP8 W8A8 executes on SM120 CUTLASS, but **fails the frozen numerical
acceptance gates** (maximum score difference 0.04249 versus the 0.03 budget).
Keep BF16 as the validated precision. FP8 remains experimental even though
average labeled ranking metrics improved on the small heldout set.

## Install

Tested target: Linux x86_64, Python 3.12, vLLM 0.33.0, one RTX 5090.
The lock file pins the actual environment, including Torch 2.14.0/CUDA 13.0.
From a source checkout:

```bash
source scripts/env.sh
uv sync --locked
```

This installs the package in editable mode and includes development tools.
Temporary files, outputs and compiler caches stay under ignored `.local/`;
source `scripts/env.sh` in each new shell. The existing Hugging Face model cache
is reused. No dependency on the separate `lmeb` project is required.

For a wheel, first provision the compatible vLLM environment, then install the
plugin without replacing its GPU dependencies:

```bash
uv build --out-dir .local/output/dist
uv pip install --no-deps .local/output/dist/jina_reranker_v35_vllm-0.1.0-py3-none-any.whl
```

`--no-deps` assumes the dependencies have already been installed from the lock
or an equivalent compatible environment. It is not a complete environment setup.

Model and tokenizer must use revision
`e8a93f33f0b22108f8c2364f8484ce3422552fbc`.
`model-reference.json` records file hashes. Obtain the full snapshot, including
its tokenizer and reference code, using `hf download` with that revision.
Check the model's own license before deployment or redistribution.

## Serve

From the source checkout, the launcher finds the fixed snapshot in the local HF
cache. Set `JINA_MODEL_DIR` if using a copied snapshot. It never downloads weights
or edits their configuration.

```bash
source scripts/env.sh
scripts/serve.sh bf16 --port 8000
```

The launcher invokes native `vllm serve`: explicit `JinaRerankerV35` architecture,
opt-in general/endpoint plugins, pooling runner, BF16 KV, FA2, mixed sliding/global
attention, chunked prefill and disabled prefix caching. It binds localhost;
additional vLLM arguments can override deployment settings. Its batch-token and
sequence limits are starting values, not a universal optimum.

After stopping BF16, experiment with FP8 using the same launcher:

```bash
scripts/serve.sh fp8 --port 8000
```

This adds native `fp8_per_channel` W8A8 and CUTLASS. All 112 backbone linears
are quantized; embedding and projector stay BF16. No calibration dataset or
separate quantized checkpoint is needed. The numerical gate failure above still
applies.

For wheel-only deployments, invoke `vllm serve` directly with
`VLLM_PLUGINS=jina_reranker_v35` and these essential settings:

```bash
export VLLM_PLUGINS=jina_reranker_v35
vllm serve "$JINA_MODEL_DIR" \
  --host 127.0.0.1 --served-model-name jina35 --model-impl vllm --runner pooling \
  --hf-overrides '{"architectures":["JinaRerankerV35"],"embedding_size":512}' \
  --pooler-config '{"task":"token_embed"}' \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
  --max-model-len 131072 --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --disable-flashinfer-autotune \
  --enable-chunked-prefill --no-enable-prefix-caching
```

## Requests

Native `/rerank`, `/v1/rerank`, `/v2/rerank`, `/score` and `/v1/score` are used
without a separate protocol:

```bash
curl http://127.0.0.1:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"jina35","query":"How can inference latency be reduced?",
       "documents":["Use optimized GPU kernels and batching.","Garden plants need water."],
       "top_n":0}'
```

For `/score`, use `text_1`/`text_2` (or the native data/queries aliases).
1:N shares the same listwise context as rerank; N:N evaluates each pair
independently, including repeated query strings. N:1 and mismatched lists are
rejected. Rerank returns original text/index, stable descending scores and native
usage. `top_n` omitted or zero returns all documents; `/score` preserves input
order. Scores are raw cosine, not probabilities.

The official profile always right-truncates queries to 1024 tokens and documents
to 8192 before formatting. Request limits accept only 0 or those exact values;
0 adds no extra request-level limit. Empty strings are allowed. Nonempty
instructions, template changes, final-prompt truncation, padding to max length,
activation changes and dynamic LoRA are unsupported. Final prompts exceeding the
engine limit or violating marker counts fail before any GPU submission. Reducing
`max_model_len` rejects oversized official blocks rather than silently repartitioning
them. Long lists split at the official count/capacity boundaries, then fuse raw
query vectors across blocks.

The plugin affects only instances explicitly selecting `JinaRerankerV35`.
Other architectures retain native behavior. vLLM internals are version-pinned;
upgrades require fresh contract and GPU validation. This does not extend native
`LLM.score()`; offline tools use `LLM.encode(..., pooling_task="token_embed")`
and the same `prepare_request`/`merge_request` functions. Generic pooling endpoints
alone do not implement the high-level rerank algorithm.

## Validation

```bash
source scripts/env.sh
uv run ruff check .
uv run ruff format --check .
uv run shellcheck -s bash scripts/*.sh
uv run pytest
```

CPU oracle tests need the fixed local model snapshot. GPU acceptance is separate:
`tools.make_acceptance_workload`, `tools.compare_reference`,
`tools.evaluate_acceptance`, `tools.check_http` and `tools.bench_serving` provide
reproducible commands with `--help`. HF reference execution uses the unmodified
official model with independent PyTorch FlexAttention; native serving uses FA2.
The trusted offline inspection tool enables vLLM's pickle-based worker callable
transport for `apply_model`; the server launcher does not enable it.

`acceptance.json` freezes numerical and engineering quality budgets. Run
`JINA_GPU_ARTIFACTS=<receipt-directory> uv run pytest -m gpu` to check actual
execution receipts, including the currently failing FP8 numerical gate. Such
receipts, benchmarks and real workload text are intentionally local artifacts.
The small labeled corpus measures ranking differences, not downstream answer
quality or a universal business guarantee.

Documents/reports live in ignored `docs/` and are absent from a fresh clone.
The local acceptance report is `docs/acceptance-report.md`; coding-agent workflow
and constraints are in [AGENTS.md](AGENTS.md).

Local acceptance results (36 real queries, including 27 heldout, plus four
semantic probes):

| Check | Result |
| --- | --- |
| BF16 vs official HF max score error | 0.00786; passes 0.01 budget |
| FP8 vs BF16 max score error | 0.04249; fails 0.03 budget |
| FP8 projected-vector error | Also exceeds frozen max/p95 budgets |
| Heldout nDCG@10, BF16 / FP8 | 0.7409 / 0.7508 |
| Heldout MRR@10, BF16 / FP8 | 0.7994 / 0.8179 |
| Native HTTP routes and real disconnect | Pass in both precisions |
| Warm HTTP p50 speedup, geometric mean | 1.25× across 9 workload/concurrency cells |

The latency cells use short lists, long lists and longer documents at concurrency
1/4/8, with 12 measured requests per cell. FP8's short-list concurrency-1 p50 was
slower (13.5 vs 12.3 ms); gains depend on workload. These are local descriptive
measurements, not production latency guarantees. Startup/compilation is excluded.

A minimal offline receipt can be produced from the built-in semantic workload:

```bash
uv run python -m tools.make_acceptance_workload --output .local/output/semantic.jsonl
uv run python -m tools.compare_reference --mode bf16 \
  --workload .local/output/semantic.jsonl --output .local/output/bf16.pt
```

Workload JSONL rows contain `id`, `query`, and `documents` (strings). Labeled
acceptance additionally uses parallel integer `relevance` labels and a `split`
of `calibration` or `heldout`. The built-in workload includes a long capacity
probe; the local real-text workload stays outside Git. Current tests: 92 CPU
checks pass; the six GPU receipt checks have five passes and one explicit FP8
numerical-gate failure.
