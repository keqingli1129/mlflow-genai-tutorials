# Quickstart: Validate MLflow Observability for Minimal RAG Chain

## Prerequisites

- Python environment synced: `uv sync`
- `.env` created from `env_template` and populated
- Ollama running locally at `http://localhost:11434`
- Chroma store pre-populated at `./chroma_db`
- MLflow server running locally on `http://127.0.0.1:5000`

## 1) Start MLflow server

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

## 2) Execute the instrumented chain once

```powershell
uv run python "RAG Chain.py"
```

Expected outcomes:
- one new run appears under experiment `minimal-rag-chain`
- run contains all required telemetry fields from [observability-contract.md](./contracts/observability-contract.md)
- run trace includes stage spans: retrieval, prompt_composition, model_invocation, response_formatting

## 3) Execute 20-question validation batch

Use the planned validation script flow (or temporary loop) to invoke 20 questions sequentially.

Expected outcomes:
- SC-001: every successful invocation has a corresponding run
- SC-002: >=95% runs include complete stage coverage
- SC-003: slowest 3 runs identifiable via `latency_total_ms`
- SC-004: failed or low-quality runs contain retrieval/timing diagnostics for root-cause analysis

## 4) Manual walkthrough validation for SC-005

- Have reviewers inspect runs/traces in MLflow UI without opening source files.
- Record whether each reviewer can explain execution flow from telemetry alone.

Expected outcomes:
- >=90% reviewers report understanding of chain flow from observability data.

## Troubleshooting

- If tracking URI is unreachable:
  - confirm server command is running and port 5000 is open
  - verify `MLFLOW_TRACKING_URI` in `.env`
- If no traces appear:
  - ensure `mlflow.langchain.autolog()` is enabled before chain invocation
- If stage timing missing:
  - verify manual stage span instrumentation for each required stage

## References

- Plan: [plan.md](./plan.md)
- Data model: [data-model.md](./data-model.md)
- Contract: [observability-contract.md](./contracts/observability-contract.md)
