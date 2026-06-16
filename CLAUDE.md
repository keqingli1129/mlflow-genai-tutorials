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

## Common Pitfalls

- **`Correctness` scorer requires `expected_facts`** in dataset `expectations` field
- **RAG judges** (`RetrievalGroundedness`, `RetrievalRelevance`) require `predict_fn` with tracing enabled — they read from the trace, not from `outputs`
- **Old vs new API**: Use `mlflow.genai.evaluate()` (MLflow 3.x), not `mlflow.evaluate()` with `model_type="databricks-agent"` (MLflow 2.x)
- **Judge model format for LiteLLM**: `"databricks/model-name"`, `"anthropic/claude-..."`, `"azure:/gpt-4o"` — not raw model names

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
