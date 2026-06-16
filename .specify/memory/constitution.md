<!--
  Sync Impact Report
  ==================
  Version change: 0.0.0 → 1.0.0 (initial ratification)
  Modified principles: N/A (first version)
  Added sections:
    - Core Principles (5 principles)
    - Technology Constraints
    - Development Workflow
    - Governance
  Removed sections: N/A
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ (Constitution Check section
      will reference these principles at plan time)
    - .specify/templates/spec-template.md ✅ (no changes needed, generic)
    - .specify/templates/tasks-template.md ✅ (no changes needed, generic)
  Follow-up TODOs: None
-->

# MLflow GenAI Tutorials Constitution

## Core Principles

### I. Reproducibility

All notebooks and scripts MUST produce consistent results given the same
environment and inputs.

- Every notebook MUST declare its MLflow experiment name at the top and
  set it explicitly via `mlflow.set_experiment()`.
- Random seeds MUST be set where stochastic behavior exists (model calls
  excluded; document non-determinism inline).
- Environment variables MUST be loaded from `.env` using `python-dotenv`;
  hard-coded credentials or endpoints are forbidden.
- The `env_template` file MUST enumerate every required variable with a
  placeholder value and a comment explaining its purpose.
- MLflow tracking URI MUST be configured via environment, never assumed.

**Rationale**: Tutorials are educational artifacts. A learner following
the sequence must get predictable outcomes without hidden state.

### II. Dependency Management

A single source of truth for dependencies MUST exist in `pyproject.toml`
managed by UV.

- All runtime dependencies MUST be declared in `pyproject.toml` under
  `[project.dependencies]` with minimum version pins (e.g., `>=3.11.1`).
- `requirements.txt` MUST NOT diverge from `pyproject.toml`; if retained
  for compatibility, it MUST be generated from the canonical source.
- `uv sync` MUST be the sole installation command documented for
  contributors; ad-hoc `pip install` in notebooks is forbidden.
- Python version MUST be pinned to `>=3.11` in `requires-python`.
- New dependencies MUST be justified in the commit message and MUST NOT
  introduce conflicting version constraints with existing packages.

**Rationale**: Reproducibility begins with deterministic dependency
resolution. A single lock mechanism prevents "works on my machine."

### III. Testing Discipline

Every reusable module in `utils/` and standalone scripts MUST have
corresponding automated tests.

- New functions added to `utils/` MUST include a pytest test in `test/`.
- Tests MUST be runnable offline where possible (mock external API calls).
- Smoke tests for external endpoints (e.g., AI Gateway) MUST be clearly
  separated from unit tests and marked with `@pytest.mark.integration`.
- Notebooks are exempt from unit testing but MUST be validated by running
  top-to-bottom without errors as a CI gate (using `nbconvert --execute`
  or equivalent) when API keys are available.
- Test commands: `uv run pytest test/` for unit tests.

**Rationale**: The project currently lacks automated tests. Incremental
adoption starts with utility modules that are shared across notebooks.

### IV. Notebook-to-Module Migration

Reusable logic MUST live in importable Python modules, not inline in
notebook cells.

- Helper functions used by more than one notebook MUST be extracted to
  `utils/` or a domain-specific module.
- Notebooks MUST import shared logic rather than duplicating it.
- Each notebook's first code cell MUST contain only imports and
  environment setup (no business logic definitions).
- When migrating code from a notebook to a module, the notebook cell
  MUST be replaced with an import statement in the same PR.
- Module public APIs MUST include type hints on function signatures.

**Rationale**: Notebooks are presentation layer. Extracting logic into
modules enables testing, reuse, and reduces drift between tutorials.

### V. Incremental Refactoring

Changes MUST be small, reversible, and independently verifiable.

- Each PR MUST address a single concern (one refactor, one feature, one
  fix). Mixed-purpose PRs MUST be split.
- Refactoring MUST NOT change observable behavior; pair every refactor
  with a test that proves equivalence.
- Large migrations (e.g., API version upgrades) MUST be broken into
  phases with intermediate working states.
- Deprecated patterns MUST be annotated with `# DEPRECATED: migrate to
  <new_pattern> by <date/version>` before removal.
- Notebook renumbering or restructuring MUST preserve git history via
  `git mv` where possible.

**Rationale**: A tutorial repo is a living document consumed by learners.
Breaking changes to notebook flow disrupt the learning path.

## Technology Constraints

- **Runtime**: Python >=3.11, managed by UV
- **ML Platform**: MLflow >=3.11.1 (3.x API exclusively; `mlflow.genai`
  namespace for evaluation, not legacy `mlflow.evaluate()`)
- **Frameworks**: LangChain >=1.2.7, LangGraph >=0.3, CrewAI >=0.108,
  LlamaIndex >=0.14, OpenAI SDK >=1.0
- **Notebook runtime**: Jupyter / JupyterLab with ipykernel
- **Provider abstraction**: `utils/clnt_utils.py` is the single factory
  for all LLM clients; notebooks MUST NOT instantiate clients directly
- **Secrets**: `.env` file (gitignored), loaded via `python-dotenv`
- **Tracking backend**: Local SQLite (`mlflow.db`) for development;
  artifacts in `mlartifacts/` (gitignored)

## Development Workflow

1. **Setup**: `uv sync` then `cp env_template .env` and fill values.
2. **Develop**: Edit notebooks or modules; run `uv run jupyter notebook`.
3. **Validate**: `uv run pytest test/` for modules; run notebook
   top-to-bottom for notebook changes.
4. **Track**: `uv run mlflow ui --port 5000` to inspect experiment runs.
5. **Commit**: Atomic commits with conventional message prefixes
   (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
6. **Review**: PRs MUST pass linting (if configured) and test suite
   before merge.

## Governance

- This constitution supersedes informal conventions. All contributions
  MUST comply; reviewers MUST verify adherence.
- Amendments require a PR updating this file with rationale in the
  commit message. Version MUST be incremented per semver rules:
  - MAJOR: Principle removal or incompatible redefinition
  - MINOR: New principle or material expansion
  - PATCH: Clarification or wording fix
- Compliance review: quarterly scan of `utils/` test coverage and
  notebook execution status.
- Runtime development guidance lives in `CLAUDE.md` and
  `.github/copilot-instructions.md`.

**Version**: 1.0.0 | **Ratified**: 2026-06-16 | **Last Amended**: 2026-06-16
