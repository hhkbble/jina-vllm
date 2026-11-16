# jina-vllm

Serve Jina Reranker v3.5 and Jina Embeddings v5 Text Small with vLLM 0.33.0.
Reranker v3.5 uses the `JinaRerankerV35` plugin for listwise scoring and hybrid
attention. Text Small works as a native vLLM embedding model and also has a
`JinaEmbeddingsV5TextSmall` plugin architecture for pool-free prefill and YaRN.

BF16 with FlashAttention 2 (FA2) is the baseline. FP8 Linear and FP8 prefill
are experimental and can change reranking scores and rankings or embedding
vectors. Test quantized configurations on your own workload before deployment.

## Install

The tested target is Linux x86_64 with one NVIDIA RTX 5090 and tensor and
pipeline parallelism both set to 1. The lock pins Python 3.12 dependencies,
including vLLM 0.33.0, Torch 2.14.0 and CUDA 13.0 packages. Use a compatible
NVIDIA driver. From the checkout, with `uv` installed, run

```bash
source scripts/env.sh
uv sync --locked --no-dev
source .venv/bin/activate
```

On Linux, long checkout paths can exceed the native engine's Unix socket path
limit. Use a shorter checkout path or point `VLLM_RPC_BASE_PATH` to a short,
writable, private directory before starting vLLM.

You can also install the plugin wheel into an existing environment with the
same vLLM and compatible GPU dependencies. Build it from the checkout and
run the install command in the target environment.

```bash
uv build --wheel --out-dir .local/output/dist
uv pip install --no-deps .local/output/dist/jina_vllm-0.1.0-py3-none-any.whl
```

## Model checkpoints

Use the same pinned snapshot for weights, config and tokenizer. Set
`JINA_MODEL_DIR` and `JINA_EMBED_MODEL_DIR` to your existing snapshots, or use
the Hugging Face CLI to populate its cache and print the snapshot paths.

```bash
JINA_MODEL_DIR=$(hf download jinaai/jina-reranker-v3.5 \
  --revision e8a93f33f0b22108f8c2364f8484ce3422552fbc --quiet)
JINA_EMBED_MODEL_DIR=$(hf download jinaai/jina-embeddings-v5-text-small \
  --revision dd76d535f5447ca3897a9c893fb1e612ead98192 --quiet)
export JINA_MODEL_DIR JINA_EMBED_MODEL_DIR
```

Text Small uses the full parent snapshot, including its `adapters/` directory.
Its config class comes from the pinned snapshot via `--trust-remote-code`.

## Start a server

Define the common Bash arrays in the shell where you will start the server.
The examples bind to `127.0.0.1:8000`. Stop the current server before running
another example on that port.

```bash
common=(
  --host 127.0.0.1 --port 8000 --model-impl vllm --runner pooling
  --dtype bfloat16 --max-num-seqs 16 --gpu-memory-utilization 0.75
  --no-enable-prefix-caching --disable-flashinfer-autotune
)
fa2=(
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'
  --kv-cache-dtype bfloat16
)
paged=(--enable-chunked-prefill --max-num-batched-tokens 8192)
```

### Reranker v3.5

```bash
reranker_hf='{"architectures":["JinaRerankerV35"],"embedding_size":512}'
reranker=(
  "$JINA_MODEL_DIR" "${common[@]}" --max-model-len 131072
  --hf-overrides "$reranker_hf" --pooler-config '{"task":"token_embed"}'
)
VLLM_PLUGINS=jina_vllm vllm serve "${reranker[@]}" "${fa2[@]}" "${paged[@]}"
```

```bash
curl --fail-with-body http://127.0.0.1:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"What causes ocean tides?","documents":["The Moon pulls on Earth’s oceans.","Wind generates surface waves."],"top_n":0}'
```

- Use `/rerank`, `/v1/rerank` or `/v2/rerank` for a query and a list of plain
  strings. Results retain the original text and index, sort by descending
  score, and preserve input order for ties. Omit `top_n` or set it to zero to
  return every document.
- Use `/score` or `/v1/score` with `text_1` and `text_2` or their native aliases.
  A single query against a document list uses listwise scoring. Equal-length
  lists score each pair independently and return scores in input order. A
  multi-query list against one document and unequal pairwise lists are rejected.
- Omit `model` on a single-model server, or pass its served model name.
  Scores are raw cosine similarities. The reranker accepts plain text including
  empty strings. It rejects template overrides, nonempty instructions, final
  prompt truncation, padding to max length, activation changes and dynamic LoRA.
- The reranker right-truncates each query to 1,024 tokens and each document
  to 8,192 tokens before formatting. The corresponding request fields accept
  zero for the checkpoint defaults or the exact limits.
- Each block holds at most 125 documents by default, subject to the engine's
  effective token limit for the formatted prompt. Documents keep input order.
  An oversized single document block produces HTTP 400 before scheduling.
  The reranker fuses query vectors across blocks in each listwise group.
  To change the document limit, add
  `--additional-config '{"jina_vllm":{"reranker_v35":{"block_size":300}}}'`.

Offline reranking uses `prepare_request` and `merge_request` from
`jina_vllm.reranker_v35.ranking` around native
`LLM.encode(..., pooling_task="token_embed")`. The same startup config belongs
in `LLM(additional_config=...)`. Native `LLM.score()` does not implement this
listwise algorithm.

### Embeddings v5 Text Small

```bash
embedding_hf='{"jina_task":"retrieval","matryoshka_dimensions":[32,64,128,256,512,768,1024]}'
embedding=(
  "$JINA_EMBED_MODEL_DIR" "${common[@]}" --max-model-len 32768 --trust-remote-code
  --pooler-config '{"task":"embed","seq_pooling_type":"LAST","use_activation":true}'
)
VLLM_PLUGINS='' vllm serve "${embedding[@]}" --hf-overrides "$embedding_hf" \
  "${fa2[@]}" "${paged[@]}"
```

vLLM merges one task adapter at startup. Use `retrieval` for search, or set
`jina_task` to `text-matching`, `classification` or `clustering`.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["Query: What causes tides?","Document: The Moon pulls on Earth’s oceans."],"dimensions":32}'
```

For retrieval, add the literal `Query: ` or `Document: ` prefix to each input.
The 32,768-token limit includes the prefix. Omit `dimensions` for 1024 dimensions,
or request 32, 64, 128, 256, 512, 768 or 1024. Native LAST pooling selects the
last token, slices to the requested width and L2-normalizes the result.
Small BF16 differences can occur across GPU batch shapes.

Offline, use native `LLM` with the same model and pooler settings and call
`llm.embed(inputs, pooling_params=PoolingParams(dimensions=32))`.

## Performance options

### Pool-free FA2

Pool-free prefill processes complete prompts in one scheduler step and
allocates no physical KV cache. It requires a Jina plugin architecture,
CUDA Model Runner V2, explicit FA2 with BF16 activations and KV dtype, and
disabled chunked prefill and prefix caching. Set both token budgets to at
least the effective model length. The 131,072-token budget below also fits
four full-length Text Small inputs in one step. The startup log shows the
selected mode, the zero KV allocation, or the reason for paged fallback.

```bash
export VLLM_PLUGINS=jina_vllm VLLM_USE_V2_MODEL_RUNNER=1
pool_free=(
  --no-enable-chunked-prefill --max-num-batched-tokens 131072
  --max-num-scheduled-tokens 131072 --long-prefill-token-threshold 0
)
embedding_plugin_hf='{"architectures":["JinaEmbeddingsV5TextSmall"],"jina_task":"retrieval","matryoshka_dimensions":[32,64,128,256,512,768,1024]}'
embedding_plugin=("${embedding[@]}" --hf-overrides "$embedding_plugin_hf")

# Choose the array for the model you want to serve.
model=("${reranker[@]}")
# model=("${embedding_plugin[@]}")
vllm serve "${model[@]}" "${pool_free[@]}" "${fa2[@]}"
```

### Pool-free FP8 prefill

For either plugin architecture, replace the FA2 options with the explicit
FP8 backend and one of its cache dtype selectors.

```bash
vllm serve "${model[@]}" "${pool_free[@]}" \
  --attention-backend JINA_FP8_PREFILL --kv-cache-dtype fp8_per_token_head_p_row
```

Use `fp8_per_token_head_p_tile` for tile scaling or `fp8_per_block` for
sequence-local block scaling. All three modes take BF16 Q/K/V, use E4M3
quantization and write BF16 output with zero physical KV allocation.
Tile scaling shares a scale across the KV tile, so even a masked column in
the same tile can affect rounding. Block scaling shares one scale per
128-token block and head for each of Q, K and V. A future token in the
same block can affect rounding through that scale. Block scaling supports
full causal attention and the 1,024-token causal window.

With `fp8_per_block`, each GPU worker tunes five token buckets (1,024 through
16,384 in powers of two) for each mask at startup. Each key uses a single
sequence and 72 Triton candidates. Allow extra startup time on an empty
cache. Restarting with matching results in Triton's cache
(`TRITON_CACHE_DIR` when set) reuses the selected configurations. Requests
launch the selected kernel directly with no benchmarking. Mixed batches
use the bucket for their actual total tokens rounded up. Totals greater than
16,384 reuse that bucket's configuration while processing their full lengths.

For a local cold and cached startup check, run the same block-mode serve
command twice with the same initially empty `TRITON_CACHE_DIR` and
`TRITON_PRINT_AUTOTUNING=1`. To check piecewise graph replay with the short
rerank request above, append both arguments when starting the loopback server.

```bash
--compilation-config '{"cudagraph_capture_sizes":[256]}'
--profiler-config '{"profiler":"torch","torch_profiler_dir":".local/output/fp8-block-trace"}'
```

Send the request once after startup, then bracket a repeat request with
`POST /start_profile` and `POST /stop_profile`. Look for `cudaGraphLaunch` in the
trace.

This backend was runtime-tested on RTX 5090.
It accepts batched and scheduled budgets of up to 524,288 tokens per worker
step and rejects incompatible configurations at startup. Each FP8 mode
exceeded numerical error thresholds in a frozen reranker comparison
against BF16 FA2.

### W8A8 Linear

Append `--quantization fp8_per_channel --linear-backend cutlass` to any serve
example to use online FP8 weights and dynamic per-token activations. For the
reranker, also append
`--quantization-config '{"ignore":["model.embed_tokens","projector.*"]}'`
to keep its embedding table and projector in BF16. The reranker requires
BF16 activations with either Linear mode. Text Small merges the selected
adapter before quantizing and keeps its embedding table in BF16 and pooling
head in FP32. Linear and attention precision are independent.

### W4A4 Linear

On SM120, the reranker can use Humming online W4A4 with pool-free
`fp8_per_block` attention. Set the two schemas below and add
`--quantization humming` to the block-mode serve command.

```bash
export VLLM_HUMMING_ONLINE_QUANT_CONFIG='{"dtype":"float4e2m1","group_size":16,"scale_dtype":"float8e4m3","scale_2_type":"channel","ignore":["model.embed_tokens","projector"]}'
export VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float4e2m1","group_size":16,"scale_dtype":"float8e4m3","quant_mode":"dynamic_group_token","allow_fallback":false}'
```

Humming packs FP4 weights as the BF16 checkpoint loads. The recipe uses
E4M3 scales for groups of 16, a second weight scale per output channel, and
a dynamic FP32 activation scale per token. The embedding table and marker
projector use BF16. Treat the recipe as experimental pending retrieval-quality
evaluation.

### CUDA allocator

`source scripts/env.sh` defaults to PyTorch's native CUDA allocator with
expandable segments. Set the same option in the launch environment for wheel
deployments that skip the script.

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
```

Both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` accept a complete
allocator configuration. The script preserves either variable, including
an explicitly empty setting. The pinned Torch build gives
`PYTORCH_CUDA_ALLOC_CONF` precedence when both are set. Use
`expandable_segments:False` in the effective variable and restart the process
to disable expandable segments.

### Idle GPU cache trim

Add `--additional-config '{"jina_vllm":{"idle_trim":0.1}}'` to a pool-free
serve command to release unused PyTorch cache when the engine is idle.
Combine plugin settings in one `jina_vllm` object if using `block_size` too.

The default `0` disables trim. A value in `(0, 1)` sets the threshold for
reserved minus allocated bytes as a fraction of total GPU memory. Trim
requires one visible CUDA device, the native PyTorch allocator with caching
enabled, the default multiprocess GPU worker and synchronous scheduling.
Leave vLLM's CuMem allocator disabled when using idle trim.
Requests already queued in EngineCore take priority. Requests arriving after
the trim commits wait for it to finish, and the next request may need to
rebuild the allocator cache. Logs show the threshold and bytes released.

## RoPE and compilation

Both plugin architectures accept native `rope_parameters` overrides. Keep
the checkpoint's theta and original context. For example, replace
`--hf-overrides` in the reranker command to use 2× YaRN and a longer limit.

```bash
reranker_yarn='{"architectures":["JinaRerankerV35"],"embedding_size":512,"max_position_embeddings":262144,"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"original_max_position_embeddings":131072,"factor":2.0}}'
vllm serve "${reranker[@]}" "${pool_free[@]}" "${fa2[@]}" \
  --hf-overrides "$reranker_yarn" --max-model-len 131073 \
  --max-num-batched-tokens 131073 --max-num-scheduled-tokens 131073
```

For Text Small, add the YaRN fields to `embedding_plugin_hf` using
`rope_theta: 3500000`, `original_max_position_embeddings: 32768` and, for 2×
scaling, `max_position_embeddings: 65536`. Set `--max-model-len` separately to
choose the serving limit. Omit the max-position override to run YaRN at the
checkpoint's service limit, or omit `rope_parameters` for checkpoint RoPE.
Extending the context can change output quality.

The plugin architectures default to QK Norm + RoPE fusion on supported vLLM
compilation platforms. Eager mode and `-rotary_embedding` disable automatic
enablement. Use
`--compilation-config '{"pass_config":{"enable_qk_norm_rope_fusion":false}}'`
to turn it off explicitly. The reranker also defaults activation-quant and
norm-quant fusion to off.

Both plugins select the native CUDA SwiGLU op for Inductor batches above the
packed gate/up indexing limit. The first batch size that needs it is 349,526
tokens for both pinned checkpoints. This applies to the total token budget
per worker step. Above that limit, compiled CUDA configs reject an explicit
`-silu_and_mul` override.
