# Development guide

## Setup and scope

From the repository root, source `scripts/env.sh` and use the pinned Python
with `uv sync --locked`. Run tools with `uv run`. Write temporary files to
`.local/tmp/` and generated outputs to `.local/output/`. Reuse the pinned
Hugging Face snapshots in [README.md](README.md#model-checkpoints) and verify
the reranker reference code against [model-reference.json](model-reference.json).
Keep dependencies and validation self-contained.

Keep required design contracts and reproducible commands in tracked files.
Optional local research, decisions and historical reports may be available
under `docs/`. Keep those files and model snapshots intact.

## Integration contracts

- **Plugin.** Keep the vLLM 0.33.0 integration out of tree. `plugin.py`
  registers separate model names and installs the scoring adapter only for
  `JinaRerankerV35`. Keep registration lightweight. Preserve native routes,
  lengths, scheduler behavior and unrelated architectures. Use child loggers
  under `vllm.jina_vllm` for worker and service diagnostics.
- **Reranker.** Keep `ranking.py` shared by offline and native HTTP scoring.
  Preserve checkpoint truncation, marker placement, raw vectors and listwise
  fusion. Plan complete prompts with the engine's effective length. Keep block
  results in input order and drain request producers before aborting their
  owned engine IDs.
- **Text Small.** Reuse the native adapter loader, embedding routes and LAST
  pooler, including native `dimensions`. Compare requested output widths at
  identical token IDs and batch shapes when diagnosing numerical changes.
- **Pool-free prefill.** Keep the worker's zero-KV spec, full-prompt scheduling
  and attention forward in agreement. Ineligible FA2 configurations retain
  paged KV. Explicit FP8 mode/backend combinations fail early when unsupported.
  Validate mixed-length batches, window boundaries and piecewise graph replay.
- **Idle trim.** Preserve priority for requests queued before trim commits.
  Verify ingress on both sides of commitment and record released bytes and
  first-request latency. Set each allocator comparison arm explicitly using
  the [CUDA allocator settings](README.md#cuda-allocator).

Follow the pinned upstream [Ruff rules][ruff] and [pre-commit checks][hooks] as
adapted in `pyproject.toml` and `.pre-commit-config.yaml`. Recheck the pinned
vLLM contracts when upgrading. Keep tests focused on changed behavior and
specific regressions. Delegate narrowly scoped research when the owning
source leaves an important design decision uncertain.

## Validation

Run the applicable checks before committing each coherent milestone.

```bash
source scripts/env.sh
uv run ruff check .
uv run ruff format --check .
uv run shellcheck -s bash scripts/*.sh
uv run pytest
```

The CPU reranker oracle needs the pinned local snapshot. For FP8 prefill
changes, run the independent CUDA oracle with
`JINA_RUN_FP8_PREFILL_TESTS=1 uv run pytest tests/test_fp8_prefill_gpu.py`.

Collect native model and kernel receipts in a fresh output directory. Close
other test engines first. The following covers BF16 and W8A8 paged FA2 on
RTX 5090 and uses the full 131,072-token service limit.

```bash
run_dir=$(mktemp -d "$JINA_VLLM_OUTPUT_DIR/reranker-probes.XXXXXX")
uv run python -m tools.make_reranker_probes --output "$run_dir/workload.jsonl"
uv run python -m tools.compare_reference --mode bf16 \
  --workload "$run_dir/workload.jsonl" --output "$run_dir/bf16.pt"
uv run python -m tools.compare_reference --mode fp8 \
  --workload "$run_dir/workload.jsonl" --output "$run_dir/fp8.pt" \
  --profile "$run_dir/fp8-trace"
JINA_GPU_ARTIFACTS="$run_dir" uv run pytest tests/test_model_gpu.py
```

Use trusted receipts. The offline tools inspect native workers via vLLM's
pickle-based callable transport. Use `tools.compare_reference --mode hf` to
collect the official checkpoint baseline on the same workload. For native
pool-free receipts, set `VLLM_USE_V2_MODEL_RUNNER=1` and pass `--pool-free`.
Add `--kv-cache-dtype` with one of the README's FP8 prefill selectors to
verify that backend.

For native HTTP verification, start a separate paged server with
`--enable-request-id-headers` and use `tools.check_http`
with its URL, probe workload and server log. The tool checks response ordering,
block counts and disconnect cleanup. For block changes, compare against
`tools.compare_reference` prompt IDs with the same token limit and
`--additional-config`. Generate `--include-count300` to exercise a higher
document limit. Workload JSONL rows contain `id`, `query` and `documents`.
Use `tools.bench_serving --help` for latency and throughput runs, with
`--nvml-device-index` to sample a physical GPU on the client host.

For idle trim, run `uv run pytest tests/test_idle_trim.py` for commitment
ordering. Use the [pool-free serve example](README.md#pool-free-fa2) with trim
off and on. Send a batch large enough to cross the configured byte threshold,
wait for the engine to drain and compare release logs and the next request's
latency with the same inputs.

For RoPE changes, run the commands below. Compare the default and YaRN prompt
IDs or hashes, worker cache sizes and outputs. The final case for each model
runs one token beyond the checkpoint limit.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
run_dir=$(mktemp -d "$JINA_VLLM_OUTPUT_DIR/yarn-probes.XXXXXX")
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

Report environment, model, HTTP, kernel and performance checks separately.
Run final model quality evaluation before release. Update README for
user-facing changes and this guide for development workflow changes.
Use portable paths in both.

[ruff]: https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/pyproject.toml
[hooks]: https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/.pre-commit-config.yaml
