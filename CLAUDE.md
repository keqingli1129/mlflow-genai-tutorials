# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A tutorial series teaching MLflow's GenAI platform (tracing, experiment tracking, prompt management, evaluation, RAG, multi-agent orchestration). Twelve sequential Jupyter notebooks covering core GenAI workflows through advanced multi-agent patterns. MLflow version: **3.11.1** (pinned in `pyproject.toml`). Requires **Python >=3.11**. Uses the **MLflow 3.x API** — specifically `mlflow.genai.evaluate()` not the old `mlflow.evaluate()`.

## Environment Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync                              # Install all dependencies
cp env_template .env                 # Create .env from template, then fill in values
uv run jupyter notebook              # Start Jupyter
uv run mlflow ui --port 5000         # Start MLflow UI (separate terminal)
```

Required `.env` variables:
- `OPENAI_API_KEY` — for notebooks using OpenAI directly
- `MLFLOW_TRACKING_URI=http://localhost:5000` — MLflow server location
- `DATABRICKS_HOST`, `DATABRICKS_TOKEN` — only if using Databricks AI Gateway
- `AI_GATEWAY_BASE_URL`, `AI_GATEWAY_MODELS` — comma-separated list of model names, Databricks AI Gateway only
- `USE_DATABRICKS_CLIENT`, `USE_DATABRICKS_AI_GATEWAY`, `USE_OPENAI_CLIENT` — set one to `"True"` to select provider in `utils/clnt_utils.py`

## Running Tests / Scripts

There are no automated unit tests. The `test/` directory contains a manual endpoint smoke test:

```bash
uv run python test/test_ai_gateway_endpoints.py
```

To run `utils/clnt_utils.py` directly (tests LangChain client connectivity):

```bash
uv run python utils/clnt_utils.py
```

Both scripts load `.env` from the project root using `python-dotenv`.

## Architecture

### Notebooks (sequential, numbered)

| Notebook | Focus |
|----------|-------|
| `01` | MLflow setup, first tracked run |
| `02` | Experiment tracking, cost tracking, parent-child runs |
| `03` | Auto-tracing with `mlflow.openai.autolog()` |
| `04` | Manual tracing: `@mlflow.trace`, `mlflow.start_span()` |
| `05` | Prompt Registry: create, version, link to experiments |
| `06` | Framework integrations: OpenAI, LangChain, LlamaIndex |
| `07` | Evaluation: built-in scorers, custom `@scorer`, DeepEval |
| `08` | Prompt optimization with GEPA algorithm |
| `09` | Complete RAG app with RAGAS evaluation |
| `10` | Multi-agent supervisor pattern with LangGraph (Genie + Knowledge Assistant) |
| `11` | LangGraph Deep Agents: planning, file system tools, sub-agent delegation |
| `12` | CrewAI multi-agent: role-based agents, hierarchical crews |

### Key MLflow 3.x Patterns Used

**Tracing:**
```python
mlflow.openai.autolog()              # Auto-trace OpenAI calls
mlflow.langchain.autolog()           # Auto-trace LangChain/LangGraph agents
mlflow.crewai.autolog()              # Auto-trace CrewAI crews
@mlflow.trace                        # Trace a function
with mlflow.start_span("name"):      # Manual span
```

**Evaluation (MLflow 3.x API):**
```python
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, Guidelines
results = mlflow.genai.evaluate(data=dataset, scorers=[...])
results = mlflow.genai.evaluate(data=dataset, predict_fn=my_fn, scorers=[...])
```

**Agent-as-a-Judge** (multi-agent notebook):
```python
from mlflow.genai.judges import make_judge
judge = make_judge(name="...", instructions="...{{ trace }}...", model="databricks/...")
feedback = judge(trace=mlflow.get_trace(trace_id))
```

### `utils/`

- **`clnt_utils.py`** — Shared client factory used across notebooks. Supports three providers controlled by env vars: OpenAI (default), Databricks workspace (`USE_DATABRICKS_CLIENT=True`), Databricks AI Gateway (`USE_DATABRICKS_AI_GATEWAY=True`). Also exports `get_langchain_chat_openai_client` and `get_databricks_langchain_chat_client` for LangChain consumers.
- **`fema_data.py`** — 200 fabricated FEMA disaster records (2020–2025) returned as a `pd.DataFrame` via `get_disaster_data()`. Used as the structured data source for the Genie subagent in notebooks 10 and 12.
- **`policy_docs.py`** — 11 synthetic FEMA policy documents returned as `dict[str, str]` via `get_policy_documents()`. Used as the knowledge base for the policy search tool in notebooks 10 and 12.

### Standalone Scripts

- **`pipeline_v1.py`** — LangChain 1.x intent-routing agent demo with Pydantic structured output, custom middleware (`PIIMiddleware`, `RoutingContextMiddleware`), and a 10-query benchmark harness. Uses `langchain.agents.create_agent` (LangChain 1.x API, not `create_react_agent`).
- **`pipeline_v1_supervisor.py`** — Multi-agent CI/CD pipeline demo (supervisor + PLANNING/CODING/TESTING/DEPLOYMENT specialists) using the agents-as-tools supervisor pattern. Instruments with `mlflow.langchain.autolog()` and logs to the `cicd-supervisor-pipeline` experiment.
- **`prompt_optimization.ipynb`** — Standalone notebook (not part of the numbered series) for exploring GEPA prompt optimization outside the structured tutorial flow.

### MLflow Tracking

Local SQLite backend (`mlflow.db`) is used by default. The `mlartifacts/` directory stores run artifacts. Both are gitignored.

## Skills (Claude Code Slash Commands)

See `SKILLS.md` for the full catalog. Key ones:

| Skill | When to use |
|-------|-------------|
| `/mlflow-onboarding` | First-time MLflow setup |
| `/instrumenting-with-mlflow-tracing` | Adding tracing to code |
| `/agent-evaluation` | Setting up/running evaluation |
| `/analyze-mlflow-trace` | Debugging a specific trace by ID |
| `/searching-mlflow-docs` | Looking up MLflow APIs (fetches live docs) |
| `/jules-agentic-activities` | Generate & publish weekly activity newsletter |

## Development Workflow: Spec-Driven Subagent Development

When asked to build a non-trivial feature in this repo, follow this pipeline rather than jumping straight to code:

1. **Shape the idea** — `superpowers:brainstorming`: conversational Q&A, propose 2-3 approaches, until scope is clear. Skip that skill's usual doc-writing/`writing-plans` terminal steps — hand the settled idea to speckit instead (steps 2-4 below replace them).
2. **Spec** — `/speckit-specify <settled idea>` → `spec.md` (speckit's own clarification-question loop resolves remaining ambiguity, max 3 questions).
   - **Testing:** explicitly ask/answer whether this feature requires automated tests — don't leave it implicit. If `.specify/memory/constitution.md` exists and its principles already mandate tests for the scope being touched, that governs and this step is a confirmation, not an open decision — `/speckit-plan`'s own Constitution Check gate is the backstop if this step misses it. Otherwise, speckit defaults to no test tasks when unstated, so this is the only point in the pipeline where "tests are needed" can still be declared from scratch.
   - **Artifact location:** check whether a PRD already exists for this project first (a prior `docs/<NNN-feature-name>/spec.md`). **If one exists, do not create a new feature directory** — add the new feature as an additional User Story (with its own priority, e.g. P2/P3) to that existing `spec.md`, updating it in place. Only when no PRD exists yet does this step create a fresh `docs/<NNN-feature-name>/` directory (via `SPECIFY_FEATURE_DIRECTORY`, since speckit's resolution order honors an explicitly-provided value over its `specs/` default) — that directory then becomes the project's one PRD for every feature after it.
   - **Isolation:** verify you're not on the default branch (`main`/`master`) before invoking. Don't assume a `before_specify` hook will create a feature branch for you — check whether one is actually configured for this project; if not, create a feature branch yourself or invoke `superpowers:using-git-worktrees` first.
   - **Modifying an existing story:** if the new feature changes or corrects what an existing User Story already promises, edit that story's requirements in place rather than adding a disconnected new one. If it's a genuinely new capability that happens to touch existing entities/behavior along the way, keep it as a new story, but its requirements must explicitly name which existing behavior it changes — so `/speckit-plan` and `/speckit-tasks` don't treat it as pure greenfield.
3. **Plan** — `/speckit-plan` → `plan.md`, `data-model.md`, `contracts/`, `quickstart.md`. **Same rule as spec:** if these already exist for the PRD, update them in place to reflect the new user story's design (extend the data model, add new contracts) — don't regenerate from scratch and discard prior architecture decisions.
4. **Tasks** — `/speckit-tasks` → `tasks.md` (phases, `T00x` IDs, `[P]` parallel markers, `[USn]` story labels). **Same rule:** if `tasks.md` already exists, append a new phase for the new user story rather than regenerating the file — regenerating would lose already-completed `[X]` checkboxes from prior work. **Cross-phase tasks:** speckit-tasks has no memory of what a prior phase already built — before accepting generated tasks for a phase that touches prior work, check each one against the existing `tasks.md`'s completed phases and the actual code, and rewrite any "create X" task as "modify existing X in `<path>`" wherever X already exists; add an explicit task to update the existing test file's assertions rather than add a new parallel test file that duplicates coverage.
5. **Execute** — `superpowers:subagent-driven-development`, adapted to speckit's task shape (its tooling expects `## Task N` headings from `writing-plans`, which `tasks.md` doesn't use — apply these adaptations manually):
   - **Phase 1 (Setup) + Phase 2 (Foundational):** one lightweight dispatch, no separate reviewer gate — this is prerequisite plumbing, not an independently reviewable deliverable.
   - **Phase 3+ (user stories):** before dispatching, read the phase once and group its `T00x` lines into dispatch units — sequential/non-`[P]` tasks that build one deliverable bundle together into a single dispatch; `[P]` tasks touching clearly separate files/components get their own dispatch. Each group: implementer subagent → reviewer subagent (spec compliance + code quality) → fix loop until clean.
   - Source the reviewer's "Global Constraints" context from `spec.md`'s functional requirements/success criteria + `plan.md`'s Constitution Check/Technical Context (+ `.specify/memory/constitution.md` if present) — not a `writing-plans` plan header, which won't exist here.
   - **Testing:** when tests are required, the implementer commits the test files it wrote during RED alongside the implementation code — they are part of the deliverable, not scratch files discarded once GREEN passes. The reviewer checks test compliance conditionally — if `spec.md`/`tasks.md` declared tests required for a group's tasks, verify the test files are present in the diff (not just quoted output in the report) and the implementer's report includes RED/GREEN evidence; if tests weren't required, raise no test finding. The reviewer does not independently demand tests beyond what was declared at `/speckit-specify` time — that's the only point where "tests are needed" gets decided (see step 2).
   - **Test file location:** follow whatever test-location convention the project already has (co-located next to source, or a mirrored `test/`/`tests/` tree). If the project has no existing convention (greenfield), default to a mirrored test tree reflecting the source structure — safer than co-location as a default since it keeps test code out of anything that gets packaged/distributed.
   - **Modifying prior-phase code:** when a group's task brief touches a file a completed phase already built, say so explicitly and name that file and its existing test file — the implementer subagent starts with zero memory of that phase and needs to read the current state first, not treat it as greenfield. The implementer updates the existing test in place, watches the new assertion go RED then GREEN, then re-runs the *full* existing test suite for that file (not just the new assertion) before committing, to catch regressions. The reviewer additionally cross-references the *original* story's requirements in `spec.md`, not just the new story's — the change must not silently break what that story already guaranteed.
   - Keep progress ledger entries keyed to speckit IDs and phase names, e.g. `Phase 3 group A (T004-T006): complete (commits <base7>..<head7>, review clean)`. Note cross-phase touches explicitly, e.g. `Phase 5 group B (T020-T022): complete — modifies Phase 3/US1 code in utils/x.py, full regression suite re-run, review clean`.
   - After all phases are done: final whole-branch review (`superpowers:requesting-code-review`), then `superpowers:finishing-a-development-branch`.

**If a prerequisite is missing when this workflow is invoked** (superpowers plugin, or spec-kit not initialized in the current project) — don't just halt and report it, ask for permission to install it, then install it: spec-kit via `uvx --from git+https://github.com/github/spec-kit.git specify init --here`; superpowers via Claude Code's plugin marketplace. Only proceed after the user says yes.

Full workflow doc: `docs/spec-driven-subagent-development-workflow.md`.

## Common Pitfalls

- **`Correctness` scorer requires `expected_facts`** in dataset `expectations` field
- **RAG judges** (`RetrievalGroundedness`, `RetrievalRelevance`) require `predict_fn` with tracing enabled — they read from the trace, not from `outputs`
- **Old vs new API**: Use `mlflow.genai.evaluate()` (MLflow 3.x), not `mlflow.evaluate()` with `model_type="databricks-agent"` (MLflow 2.x)
- **Judge model format for LiteLLM**: `"databricks/model-name"`, `"anthropic/claude-..."`, `"azure:/gpt-4o"` — not raw model names

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
