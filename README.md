# jina-reranker-v35-vllm

Independent out-of-tree vLLM integration for `jinaai/jina-reranker-v3.5`.
Current status: implementation is in progress. The package and opt-in plugin
entrypoints are installed; model and serving acceptance are pending.

Local design documents and reports live in `docs/`, which is intentionally
excluded from version control and is not included in a new clone. Coding-agent
workflow instructions live in [AGENTS.md](AGENTS.md).

## Environment

Target: Linux x86_64, Python 3.12, vLLM 0.33.0, one RTX 5090.
uv manages the project environment and dependency lock.

From the project root, run:

```bash
source scripts/env.sh
uv sync --locked
source .venv/bin/activate
```

Alternatively, after sourcing `scripts/env.sh`, use `uv run --locked <command>`.
The environment setup follows the [uv project guide](https://docs.astral.sh/uv/guides/projects/)
and [uv environment variable reference](https://docs.astral.sh/uv/reference/environment/).

`scripts/env.sh` keeps temporary files in `.local/tmp/`, transient outputs in
`.local/output/`, and tool caches and uv-managed Python installations under
`.local/`. Source it in each new shell before project commands. `.venv/` and
`.local/` are ignored by Git. Hugging Face cache settings are preserved so the
existing model download can be reused.

The design pins model and tokenizer revision
`e8a93f33f0b22108f8c2364f8484ce3422552fbc`. `model-reference.json` records the
snapshot's file sizes and SHA-256 checksums. Environment details and setup checks
are recorded locally in `docs/environment-initialization.md`.
