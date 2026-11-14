# Project instructions

## Scope and design

Read `docs/jina-vllm-reranker-v35-design.md` before implementation, architecture
changes, or validation planning. It covers the out-of-tree integration and
reranker contract. Preserve reports and design notes in ignored `docs/`.
For changes to pool-free prefill, also read `docs/pool-free-prefill.md` for
the pinned V2 runner contracts and GPU verification commands.
For `JINA_FP8_PREFILL`, see `docs/pool-free-fp8-design-plan.md` and
`docs/pool-free-fp8-acceptance.md`. Run the independent CUDA oracle with
`JINA_RUN_FP8_PREFILL_TESTS=1 uv run pytest tests/test_fp8_prefill_gpu.py`.

For authorized development, complete implementation and acceptance autonomously
without asking the user implementation questions. Preserve the user's intent; when the design is demonstrably
infeasible or unreasonable, make justified changes to implementation details and
record the evidence and decision in `docs/`. When uncertain, delegate the specific
question to a research subagent, investigate, and choose the best supported
approach before proceeding.

The earlier reranker evaluation project, `../lmeb` when available, may supply
temporary data, examples, or application context. This project is independent:
keep its dependencies, environment, and implementation self-contained.

## Environment and local files

Source `scripts/env.sh` before setup, development, or validation commands. Use uv,
the pinned Python, and the project `.venv`; restore dependencies with
`uv sync --locked`. Keep temporary files and transient outputs under `.local/`;
use `.local/tmp/` instead of the system temporary directory. Store documents and
reports in `docs/`.
Both directories remain ignored. Reuse the user's downloaded Hugging Face model
snapshot; preserve its files and revision, and avoid redundant downloads.

## Engineering standards

Follow the pinned vLLM version's coding style. Before adding implementation code,
consult its [Ruff configuration](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/pyproject.toml)
and [pre-commit checks](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/.pre-commit-config.yaml).
Adapt the applicable formatting and lint rules to this repository's layout and
run them for development milestones; revisit alignment when upgrading vLLM.

Choose simple, cohesive designs with carefully considered details. Keep
abstractions and defensive checks proportionate to concrete requirements and
failure modes. Resolve uncertain tradeoffs through the research process above.
Tests should cover changed contracts, meaningful edge cases, and regressions at
the appropriate unit or integration level. Verify observable behavior and avoid
redundant tests that merely repeat the implementation.

## Documentation roles

Keep `AGENTS.md` focused on coding-agent workflow, constraints, and validation.
Keep `README.md` focused on user-visible status, capabilities, installation,
usage, and limitations. Update each when relevant behavior or workflow changes,
in the same milestone commit; review both frequently during early development.
Use repository-relative paths and portable commands in tracked documentation.
Keep machine-specific paths and investigation details in ignored `docs/` reports.

## Commits and validation

Commit completed, coherent development milestones rather than waiting until the
end of implementation. Check the working tree and preserve unrelated changes.
Run checks appropriate to each milestone and report what was actually verified.
Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`
for code milestones; check Bash with `uv run shellcheck -s bash scripts/*.sh`.
For real GPU validation, use the native CLI and offline receipt commands in
README. Set `JINA_GPU_ARTIFACTS` to trusted native receipts for the model
structure and kernel trace checks. Run final model quality evaluation before
release.
Use a child logger under `vllm.jina_vllm` for native startup/cleanup receipts.
Report executed environment, model, HTTP, kernel and performance checks
separately.
