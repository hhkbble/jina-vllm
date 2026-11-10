# Project instructions

## Scope and design

Read `docs/jina-reranker-v35-vllm-out-of-tree-design.md` before implementation,
architecture changes, or validation planning. It defines the intended out-of-tree
vLLM integration and acceptance criteria. Documents and reports live in `docs/`,
which is deliberately ignored by Git; preserve existing local documents.

Development and acceptance are authorized. Complete the work autonomously
without asking the user implementation
questions. Preserve the user's intent; when the design is demonstrably
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
for code milestones. Real GPU acceptance is opt-in and runs separately.
Distinguish environment/import checks from model, HTTP, numerical, and performance
acceptance; only claim results supported by executed validation.
