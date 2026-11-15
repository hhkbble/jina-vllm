# jina-vllm

Jina model serving with vLLM 0.33.0. Jina Embeddings v5 Text Small uses
the native embedding model and API. Jina Reranker v3.5 uses the out-of-tree
adapter for its per-layer attention windows and listwise scoring.

BF16 is the development baseline. FP8 Linear and FP8 prefill are experimental.
Final quality evaluation is scheduled before release.

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
uv pip install --no-deps .local/output/dist/jina_vllm-0.1.0-py3-none-any.whl
```

`--no-deps` assumes the dependencies have already been installed from the lock
or an equivalent compatible environment. It is not a complete environment setup.

Use the model and tokenizer from the same local snapshot. The model-specific
revisions are listed below. `model-reference.json` records the reranker
snapshot hashes.

### CUDA allocator

`source scripts/env.sh` defaults to the PyTorch native CUDA allocator with
expandable segments. Native `vllm serve` and offline Python commands launched
from that shell inherit the setting, including their GPU workers. Both
`PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` can supply the complete
configuration. When both are set, the pinned Torch build uses
`PYTORCH_CUDA_ALLOC_CONF`. The shell script preserves either variable, including
an explicitly empty value.

For a wheel deployment or any launch that skips `scripts/env.sh`, set the
following in the launch environment before starting `vllm serve` or Python.

```bash
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True
```

To turn expandable segments off, use
`PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False` as the launch
setting, or set `expandable_segments:False` in your existing
`PYTORCH_CUDA_ALLOC_CONF`. Start a fresh process to apply the change.

### Idle GPU cache trim

The pool-free FA2 and FP8 serve commands below pass `idle_trim: 0.1` explicitly.
The plugin defaults to off. To turn it off, omit the field or pass

```bash
--additional-config '{"jina_vllm":{"idle_trim":0}}'
```

Use a number in `[0, 1)`. When the engine drains and its worker finishes GPU
work, the plugin tries to release unused PyTorch cache if reserved minus
allocated memory exceeds this fraction of the worker GPU's total memory.
Use pool-free CUDA Model Runner V2 with exactly one visible CUDA device,
the native PyTorch allocator, and the default worker, multiprocessing, and
synchronous scheduling.
Startup reports the effective byte threshold. Each trim reports the bytes released.

Requests already queued in EngineCore take priority. Requests queued after the
trim commits wait for it to finish, and the next request may need to rebuild
the cache. Put other plugin options in the same JSON object.

## Serve Jina Embeddings v5 Text Small

Use the parent checkpoint `jinaai/jina-embeddings-v5-text-small` at revision
`dd76d535f5447ca3897a9c893fb1e612ead98192`, including its `adapters/` directory.
Set `JINA_EMBED_MODEL_DIR` to this snapshot. The native loader merges one task
adapter at startup. Start with `retrieval`; the other options are
`text-matching`, `classification`, and `clustering`. Verify that the startup log
reports loading the requested adapter. The pinned config class is loaded from
the snapshot via `--trust-remote-code`.

```bash
source scripts/env.sh
source .venv/bin/activate
export JINA_EMBED_MODEL_DIR=/path/to/the/pinned/embedding/snapshot

embedding=(
  "$JINA_EMBED_MODEL_DIR" --host 127.0.0.1 --port 8000
  --runner pooling --model-impl vllm --trust-remote-code --dtype bfloat16
  --pooler-config '{"task":"embed","seq_pooling_type":"LAST","use_activation":true}'
  --max-model-len 32768 --max-num-seqs 16 --disable-flashinfer-autotune
)
embedding_hf='{"jina_task":"retrieval","matryoshka_dimensions":[32,64,128,256,512,768,1024]}'
embedding_fa2=(
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'
  --kv-cache-dtype bfloat16
)
embedding_paged=(
  --enable-chunked-prefill --max-num-batched-tokens 8192 --no-enable-prefix-caching
)
VLLM_PLUGINS='' vllm serve "${embedding[@]}" --hf-overrides "$embedding_hf" \
  "${embedding_fa2[@]}" "${embedding_paged[@]}"
```

Use the literal `Query: ` or `Document: ` prefix on each input. The 32,768-token
limit includes the prefix. Omit `dimensions` for 1024 dimensions, or request
32, 64, 128, 256, 512, 768, or 1024. Native pooling selects the last token,
slices to the requested width, and L2 normalizes the vector.

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["Query: What causes waves?","Document: Wind transfers energy to water."],"dimensions":32}'
```

Offline, construct native `LLM` with the same model and pooling configuration,
then use `llm.embed(inputs, pooling_params=PoolingParams(dimensions=32))`.
The same input can show small BF16 numerical differences across GPU batch shapes.
The `JinaEmbeddingsV5TextSmall` plugin architecture also applies the shared
[QK Norm + RoPE fusion default](#qk-norm--rope-fusion) and
[large-batch SwiGLU protection](#large-batch-swiglu).

### Text Small W8A8 Linear

Use native online FP8 per-channel weights and dynamic per-token activations.
vLLM merges the selected adapter before quantizing the backbone Linear weights.
The embedding table stays BF16 and the pooling head stays FP32.

```bash
embedding_w8a8=(--quantization fp8_per_channel --linear-backend cutlass)
VLLM_PLUGINS='' vllm serve "${embedding[@]}" --hf-overrides "$embedding_hf" \
  "${embedding_fa2[@]}" "${embedding_paged[@]}" "${embedding_w8a8[@]}"
```

### Text Small pool-free FA2

Select the plugin architecture on the same snapshot. The service keeps the
native embedding API, task adapter loader, and LAST pooler.

```bash
export VLLM_PLUGINS=jina_vllm VLLM_USE_V2_MODEL_RUNNER=1
embedding_pool_free_hf='{"architectures":["JinaEmbeddingsV5TextSmall"],"jina_task":"retrieval","matryoshka_dimensions":[32,64,128,256,512,768,1024]}'
embedding_pool_free=(
  --no-enable-chunked-prefill --no-enable-prefix-caching
  --max-num-batched-tokens 131072
  --long-prefill-token-threshold 0
)
vllm serve "${embedding[@]}" --hf-overrides "$embedding_pool_free_hf" \
  "${embedding_fa2[@]}" "${embedding_pool_free[@]}" \
  --additional-config '{"jina_vllm":{"idle_trim":0.1}}'
```

The 131,072-token batch and scheduler budgets cover four 32,768-token
inputs in one worker step. The scheduler budget inherits the batch budget.
Keep prefix caching disabled and let vLLM derive the KV allocation. Check
the startup logs for pool-free FA2 selection and zero physical KV allocation
across all 28 layers. Ineligible FA2 configurations use paged KV and log the reason.
Set `idle_trim` to `0` when using a paged KV configuration.

### Text Small pool-free FP8 prefill

Use the same pool-free settings with the explicit FP8 backend.

```bash
vllm serve "${embedding[@]}" --hf-overrides "$embedding_pool_free_hf" \
  "${embedding_pool_free[@]}" --attention-backend JINA_FP8_PREFILL \
  --kv-cache-dtype fp8_per_token_head_p_row \
  --additional-config '{"jina_vllm":{"idle_trim":0.1}}'
```

Use `fp8_per_token_head_p_tile` to select tile scaling. The backend uses FP8
attention math and BF16 Q/K/V input and output. Append
`"${embedding_w8a8[@]}"` to also select W8A8 Linear. Invalid explicit
FP8 prefill configurations fail at startup.

In tests with `--max-num-seqs 4` on RTX 5090, FA2 and both FP8 attention modes
processed four 32,768-token inputs in one worker step and returned four
normalized vectors. With BF16 Linear, worker peak allocated memory was
4.43 GiB across all three paths.
Peak reserved memory was 6.76 GiB for FA2 and 7.26 GiB for both FP8 modes.
The local measurement report is `docs/embeddings-v5-text-small-validation.md`.

### Text Small YaRN

Use the plugin architecture, the same retrieval adapter, and the native
Hugging Face override for RoPE. For example, serve inputs up to 32,769 tokens
with 2× YaRN and the existing pool-free FA2 example:

```bash
embedding_yarn='{"architectures":["JinaEmbeddingsV5TextSmall"],"jina_task":"retrieval","matryoshka_dimensions":[32,64,128,256,512,768,1024],"max_position_embeddings":65536,"rope_parameters":{"rope_type":"yarn","rope_theta":3500000,"original_max_position_embeddings":32768,"factor":2.0}}'
vllm serve "${embedding[@]}" --max-model-len 32769 --hf-overrides "$embedding_yarn" \
  "${embedding_fa2[@]}" "${embedding_pool_free[@]}"
```

To enable YaRN at the native 32,768-token service limit, omit the
`max_position_embeddings` override and keep `--max-model-len 32768`.
Omit `rope_parameters` to use the checkpoint's RoPE.

## Serve Jina Reranker v3.5

Use revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc` for this model.
Use the pinned local snapshot as the model path. From a source checkout, run
`source scripts/env.sh` and `source .venv/bin/activate` before the examples.
For a wheel deployment, activate the environment where the wheel is installed.
Start one server at a time and reuse the Bash arrays in the same shell.

```bash
export VLLM_PLUGINS=jina_vllm
export JINA_MODEL_DIR=/path/to/the/pinned/snapshot

reranker=(
  "$JINA_MODEL_DIR"
  --host 127.0.0.1 --port 8000 --model-impl vllm --runner pooling
  --hf-overrides '{"architectures":["JinaRerankerV35"],"embedding_size":512}'
  --pooler-config '{"task":"token_embed"}'
  --dtype bfloat16 --no-enable-prefix-caching --disable-flashinfer-autotune
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.75
)
fa2=(
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'
  --kv-cache-dtype bfloat16
)
paged=(--enable-chunked-prefill --max-num-batched-tokens 8192)
w8a8=(
  --quantization fp8_per_channel
  --quantization-config '{"ignore":["model.embed_tokens","projector.*"]}'
  --linear-backend cutlass
)

# Paged BF16
vllm serve "${reranker[@]}" "${fa2[@]}" "${paged[@]}"
```

Stop the BF16 server before starting paged W8A8 in the same shell.

```bash
vllm serve "${reranker[@]}" "${fa2[@]}" "${paged[@]}" "${w8a8[@]}"
```

W8A8 uses native online quantization with CUTLASS for all 112 backbone
linears. The embedding and projector stay BF16.

### Pool-free FA2 prefill

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
pool_free=(
  --no-enable-chunked-prefill --max-num-batched-tokens 131072
  --max-num-scheduled-tokens 131072 --long-prefill-token-threshold 0
)
vllm serve "${reranker[@]}" "${fa2[@]}" "${pool_free[@]}" \
  --additional-config '{"jina_vllm":{"idle_trim":0.1}}'
```

Jina v3.5 pooling on CUDA Model Runner V2 selects pool-free prefill automatically
when `FLASH_ATTN` and `flash_attn_version=2` are explicit, all layers resolve
to FA2 with BF16 Q/K/V, and both chunked prefill and prefix caching are off.
Use `--kv-cache-dtype bfloat16` or `auto` that resolves to BF16. Both the
batch and effective scheduler token budgets must cover `max-model-len`, and
any long-prefill threshold must allow the entire prompt in one step.

The startup log reports the selected mode and, in pool-free mode, confirms
zero KV cache tensors across all 28 layers. Chunked prefill,
automatic attention backend selection, quantized KV, and configurations that
require a KV pool retain native paged KV and log the reason. Set `idle_trim` to
`0` for these configurations. Native parameter validation still applies.

Pool-free prefill uses the same rerank API, sliding windows and listwise
processing. Append `"${w8a8[@]}"` to use FP8 Linear with BF16 attention.
On RTX 5090, the BF16 service ran a formatted 131,072-token
single block in one scheduler step with zero physical KV allocation. A
131,073-token block returned HTTP 400 before scheduling. The earlier 16,384-token
checks cover both Linear precisions, mixed-length batches, eager, and default
vLLM compilation with piecewise CUDA Graphs. FA2 runs outside graph capture.
In those checks, the maximum single-block score differences from same-precision
paged FA2 were 0.00160 for BF16 Linear and 0.00563 for FP8 W8A8 Linear. The
126-document, two-block FP8 case differed by up to 0.02277.
Local design notes, commands and results are in `docs/pool-free-prefill.md`.
For offline verification, pass `--pool-free` to `tools.compare_reference`
with `--mode bf16` or `--mode fp8`.

### Pool-free FP8 prefill

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve "${reranker[@]}" "${pool_free[@]}" \
  --attention-backend JINA_FP8_PREFILL \
  --kv-cache-dtype fp8_per_token_head_p_row \
  --additional-config '{"jina_vllm":{"idle_trim":0.1}}'
```

Select either `fp8_per_token_head_p_row` or `fp8_per_token_head_p_tile`.
Both values require `JINA_FP8_PREFILL`, and that backend accepts only those
two values. For offline use, pass the same strings as
`LLM(attention_backend="JINA_FP8_PREFILL", kv_cache_dtype=...)` with the
same full-prompt settings. Enable the plugin through
`VLLM_PLUGINS=jina_vllm` before constructing the LLM.
`tools.compare_reference --pool-free --kv-cache-dtype <mode>` also sets the
backend and full-context budgets.

The FP8 backend requires a supported Jina plugin architecture, NVIDIA CUDA
Model Runner V2, BF16 activations and complete prompts in one scheduler step.
It quantizes
Q/K/V per token and head for each forward, uses E4M3 for both attention dot
products and returns BF16. Both modes contribute zero physical KV cache.
Standard KV dtype values retain their native meaning. Append
`"${w8a8[@]}"` to combine this backend with W8A8 Linear.

The row mode scales the probability numerator per query row and KV tile.
The tile mode shares a scale across each KV head's tile. Causal and window
masks still zero excluded weights. In tile mode, a masked column in the same
tile can affect the rounding of the included columns through the shared scale.

FP8 attention runs between piecewise graph segments.

On RTX 5090, four offline latency cases showed a 1.12–1.18× geometric
mean speedup against same-Linear pool-free FA2. RTX 5090 (SM120) is the
runtime-tested device. The kernel also has E4M3 compiler receipts for SM89,
SM90, SM100, SM103 and SM121. Historical measurements are in
`docs/pool-free-fp8-acceptance.md`.

### Reranker YaRN

Use the checkpoint's base theta and original context in the native override.
This example configures a 2× YaRN cache and serves up to 131,073 tokens:

```bash
reranker_yarn='{"architectures":["JinaRerankerV35"],"embedding_size":512,"max_position_embeddings":262144,"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"original_max_position_embeddings":131072,"factor":2.0}}'
vllm serve "${reranker[@]}" "${fa2[@]}" "${pool_free[@]}" \
  --hf-overrides "$reranker_yarn" --max-model-len 131073 \
  --max-num-batched-tokens 131073 --max-num-scheduled-tokens 131073
```

The settings for RoPE and service length are independent. To enable YaRN with
the native 131,072-token length, omit the `max_position_embeddings` override
and use the original length and token budgets. Omit `rope_parameters` to use
the checkpoint's RoPE. Each no-pool token budget must cover the service length.
For both plugin models, FP8 prefill supports a scheduled budget of up to
524,288 tokens per worker step.

## QK Norm + RoPE fusion

The `JinaEmbeddingsV5TextSmall` and `JinaRerankerV35` plugin architectures
default to QK Norm + RoPE fusion with vLLM compilation on CUDA, ROCm and XPU.
This default enables the native rotary embedding custom op. Eager mode or
`-rotary_embedding` in `custom_ops` prevents automatic enablement. To turn
the fusion off, pass
`--compilation-config '{"pass_config":{"enable_qk_norm_rope_fusion":false}}'`
or set the same pass option in the offline `compilation_config`.

The reranker defaults activation-quant and norm-quant fusion to off.
Text Small retains vLLM's defaults for those two passes.

## Large-batch SwiGLU

Both Jina plugin architectures use vLLM's native CUDA SwiGLU when Inductor is
active and the worker's `--max-num-batched-tokens` exceeds the 32-bit indexing
limit for the model's packed gate/up projection. The pinned checkpoints both
cross this limit at 349,526 batched tokens. The budget counts the entire worker
step, including batches of shorter inputs. Above the limit, compiled CUDA
configurations reject an explicit `-silu_and_mul` override. With compilation
disabled, vLLM continues to honor the existing custom-op settings.

## Requests

Native `/rerank`, `/v1/rerank`, `/v2/rerank`, `/score` and `/v1/score` are used
through the native request schema. The optional `model` field names the
model advertised by vLLM. A server with one model can omit it.

```bash
curl http://127.0.0.1:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"How can inference latency be reduced?",
       "documents":["Use optimized GPU kernels and batching.","Garden plants need water."],
       "top_n":0}'
```

For `/score`, use `text_1`/`text_2` (or the native data/queries aliases).
1:N shares the same listwise context as rerank; N:N evaluates each pair
independently, including repeated query strings. N:1 and mismatched lists are
rejected. Rerank returns original text/index, stable descending scores and native
usage. `top_n` omitted or zero returns all documents; `/score` preserves input
order. Scores are raw cosine, not probabilities.

The reranker right-truncates queries to 1024 tokens and documents
to 8192 before formatting. Request limits accept only 0 or those exact values;
0 adds no extra request-level limit. Empty strings are allowed. Nonempty
instructions, template changes, final-prompt truncation, padding to max length,
activation changes and dynamic LoRA are unsupported.

Each rerank block holds up to 125 documents by default. The planner takes
documents in order and starts a new block when the next document would exceed
the count limit or the actual token limit for the complete prompt. The token
limit is vLLM's effective `max_model_len`. Exact counting can change block
boundaries for long requests compared with the original heuristic. If a single
document with its template exceeds the limit, the request fails before GPU
submission and reports its index and token count. Raw query vectors are fused
across blocks in the same group.

Set a positive `block_size` at startup to change the document limit. For example,
set the following option in the reranker serving command. For a pool-free
instance with idle trim enabled, keep `"idle_trim":0.1` in the same `jina_vllm`
object.

```bash
--additional-config '{"jina_vllm":{"reranker_v35":{"block_size":300}}}'
```

In Python, pass the same dictionary as `LLM(additional_config=...)`.

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

CPU oracle tests use the fixed local snapshot. Generate the four reranker
probes, then collect native BF16 and FP8 model receipts individually. The
capacity probe needs the full 131,072-token context.

```bash
run_dir=.local/output/reranker-probes
uv run python -m tools.make_reranker_probes --output "$run_dir/workload.jsonl"
uv run python -m tools.compare_reference --mode bf16 \
  --workload "$run_dir/workload.jsonl" --output "$run_dir/bf16.pt"
uv run python -m tools.compare_reference --mode fp8 \
  --workload "$run_dir/workload.jsonl" --output "$run_dir/fp8.pt" \
  --profile "$run_dir/fp8-trace"
JINA_GPU_ARTIFACTS="$run_dir" uv run pytest tests/test_model_gpu.py
JINA_RUN_FP8_PREFILL_TESTS=1 uv run pytest tests/test_fp8_prefill_gpu.py
```

Use trusted receipts for the GPU structure and kernel trace checks. The offline
inspection tool uses vLLM's pickle-based worker callable transport for
`apply_model`. It also supports collecting output from the official HF model.

`tools.check_http` checks native responses, completed block counts, and
disconnect cleanup. Run it against a dedicated paged test server started with
`--enable-request-id-headers`, and provide the probe workload and server log.
To verify a larger document limit, generate the workload with
`--include-count300` and supply the same `--additional-config` to the server,
`tools.check_http`, and `tools.compare_reference` for the corresponding checks.
`tools.bench_serving` measures latency and throughput and accepts the same
configuration for its CPU planning check.
Both commands expose their options with `--help`. Workload JSONL rows contain
`id`, `query`, and a list of `documents`. Local design notes and measurement
reports live in ignored `docs/`.

### YaRN probes

Run one offline engine at a time with the fixed checkpoints. This checks the
same reranker prompts with and without YaRN, then executes one extended block:

```bash
source scripts/env.sh
export VLLM_USE_V2_MODEL_RUNNER=1
run_dir=.local/output/yarn-probes
uv run python -m tools.make_reranker_probes --include-extended \
  --output "$run_dir/workload.jsonl"
head -n 2 "$run_dir/workload.jsonl" > "$run_dir/paired.jsonl"
tail -n 1 "$run_dir/workload.jsonl" > "$run_dir/extended.jsonl"
uv run python -m tools.compare_reference --mode bf16 --pool-free \
  --workload "$run_dir/paired.jsonl" --output "$run_dir/reranker-default.pt"
uv run python -m tools.compare_reference --mode bf16 --pool-free \
  --workload "$run_dir/paired.jsonl" --output "$run_dir/reranker-yarn.pt" \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"original_max_position_embeddings":131072,"factor":2.0}}'
uv run python -m tools.compare_reference --mode bf16 --pool-free \
  --workload "$run_dir/extended.jsonl" --output "$run_dir/reranker-extended.pt" \
  --max-model-len 131073 \
  --hf-overrides '{"max_position_embeddings":262144,"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"original_max_position_embeddings":131072,"factor":2.0}}'
for mode in default yarn extended; do
  uv run python -m tools.check_embedding_rope --mode "$mode" \
    --output "$run_dir/text-small-$mode.json"
done
```

The Text Small probe uses native `LLM.embed` and the same token IDs in the
first two modes. Compare the prompt IDs or hashes, cache rows, and outputs in
the two native-length receipts. The extended probes use BF16 pool-free FA2
with inputs of 131,073 and 32,769 tokens respectively. Both offline tools
inspect the worker using vLLM's pickle-based callable transport.
