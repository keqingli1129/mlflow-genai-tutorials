# Implementation Plan: Add MLflow Observability

**Branch**: `[001-add-mlflow-observability]` | **Date**: 2026-06-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-add-mlflow-observability/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add comprehensive MLflow observability to the existing simple RAG chain in `RAG Chain.py` without changing answer-generation behavior. The approach is a minimal, layered instrumentation pass: explicit run lifecycle management (`mlflow.start_run()`), MLflow LangChain autologging plus manual span annotations for stage-level timing, structured params/metrics/tags logging, standardized exception/status capture, and Python structured logging suitable for tutorial walkthroughs.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python >=3.11

**Primary Dependencies**: mlflow>=3.11.1, langchain>=1.2.7, langchain-chroma>=1.0.0, langchain-ollama>=1.1.0, python-dotenv>=1.0.0

**Storage**: MLflow backend store (`sqlite:///mlflow.db` local), MLflow artifacts (`./mlartifacts` local), Chroma persistent directory (`./chroma_db`)

**Testing**: Local scripted validation run with 20-question set, manual MLflow UI verification, optional pytest coverage for any new helper module in `utils/`

**Target Platform**: Local Windows/macOS/Linux development environment with local MLflow tracking server and local Ollama server

**Project Type**: Tutorial repository with standalone Python scripts and Jupyter notebooks

**Performance Goals**: 100% run creation on successful invocations; >=95% complete stage coverage in traces; per-run total latency and stage timing available for top-3 slow-run identification from 20 validation questions

**Constraints**: Instrumentation must not alter model outputs or retrieval behavior; preserve existing chain pipeline; keep changes localized to existing script and optional lightweight helper; no new infrastructure services

**Scale/Scope**: Single simple RAG chain script (`RAG Chain.py`), local run volume (tutorial usage), 20-question validation set for observability acceptance

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Pre-Design Gate Status: PASS

- Reproducibility: PASS. Plan uses `.env` loading, explicit experiment setup, and local tracking URI conventions.
- Dependency Management: PASS. No new dependency is required; implementation uses existing `mlflow` and standard library `logging/time`.
- Testing Discipline: PASS with condition. If helper logic is introduced in `utils/`, add pytest tests in `test/`; otherwise use explicit validation scenario in quickstart.
- Notebook-to-Module Migration: PASS. Scope is a standalone script; no duplicated notebook logic introduced.
- Incremental Refactoring: PASS. Change set is constrained to observability instrumentation with non-interference checks.

## Project Structure

### Documentation (this feature)

```text
specs/001-add-mlflow-observability/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
RAG Chain.py
README.md
pyproject.toml
utils/
├── __init__.py
├── clnt_utils.py
├── fema_data.py
└── policy_docs.py
test/
└── test_ai_gateway_endpoints.py
specs/
└── 001-add-mlflow-observability/
    ├── spec.md
    ├── plan.md
    ├── research.md
    ├── data-model.md
    ├── quickstart.md
    └── contracts/
```

**Structure Decision**: Keep a single-project script-first layout. Apply instrumentation directly in `RAG Chain.py` with optional extraction of reusable observability helpers into one module under `utils/` only if needed to keep script readability.

## Implementation Strategy

### 1. Experiment and Run Setup (FR-001)

- Keep `mlflow.set_experiment(EXPERIMENT_NAME)` in `main()` as the experiment naming anchor.
- Add explicit run lifecycle inside `ask()` (or a new `invoke_with_observability()` wrapper) with `with mlflow.start_run(run_name=...)` so each invocation creates exactly one run.
- Use deterministic run naming convention: `rag_chain__{model}__{YYYYMMDD_HHMMSS}` and add tags: `feature=mlflow-observability`, `chain_type=minimal-rag`, `llm_model`, `embedding_model`.
- Ensure nested run behavior is disabled to avoid accidental multiple parent runs (`nested=False` default).

### 2. End-to-End Tracing Strategy (FR-002, FR-007)

- Retain `mlflow.langchain.autolog()` to capture LangChain runnable spans automatically.
- Add manual stage spans with `mlflow.start_span()` around explicit stages not guaranteed by autolog naming:
  - `retrieval`
  - `prompt_composition`
  - `model_invocation`
  - `response_formatting`
- Keep span boundaries read-only (no data mutation) and bind span attributes for doc count, context chars, and stage duration.
- Preserve run-to-trace linkage by executing all spans within active run context.

### 3. Params, Metrics, Tags, and Artifacts (FR-003, FR-004, FR-005, FR-008)

- Params (low-cardinality): `k`, `search_type`, `collection_name`, `persist_directory`, `llm_model`, `embedding_model`.
- Tags (categorical): `component=rag-chain`, `observability_version=v1`, `tracking_mode=local`, `question_hash`.
- Metrics (numeric):
  - `latency_total_ms`
  - `latency_retrieval_ms`
  - `latency_prompt_composition_ms`
  - `latency_model_invocation_ms`
  - `latency_response_formatting_ms`
  - `retrieved_doc_count`
  - `context_char_count`
  - `answer_char_count`
- Log user question and final answer as run inputs/outputs and as a small JSON artifact (`run_diagnostics.json`) for side-by-side comparison.

### 4. Run Status and Error Recording (FR-006)

- Wrap invocation in `try/except/finally`.
- On success: set tag `run_status=success` and metric `error_flag=0`.
- On exception: set tag `run_status=failure`, metric `error_flag=1`, and tags `error_type`, `error_stage`; log exception message/traceback artifact (`error_details.txt`), then re-raise to preserve existing failure semantics.

### 5. Structured Logging (FR-010)

- Configure module-level logger with consistent format: timestamp, level, stage, run_id, short message.
- Emit stage begin/end logs with key-value payloads (doc_count, latency_ms, model name).
- Keep logging human-readable for tutorial flow while avoiding secret leakage and overly verbose raw context dumps.

### 6. Edge Case Handling

- Zero retrieval results: log `retrieved_doc_count=0`, set tag `retrieval_empty=true`, continue with fallback context string.
- Model failure/timeout: capture exception stage and status metadata, log traceback artifact, re-raise.
- Oversized prompt/output: log sizes; truncate only logged previews (not actual chain input/output) with `*_preview_truncated=true` tag.
- Unreachable MLflow URI: detect initialization/connectivity failure before invocation; degrade to console-only logging and keep chain execution path intact.

### 7. Non-Interference Guarantee (FR-009)

- Instrumentation wrappers are side-effect free relative to prompt text, retrieval query, and model invocation parameters.
- Do not transform question, retrieved docs, or final answer for model path; any truncation is logging-only.
- Preserve current chain wiring and return value contract for `ask()` and `main()`.

### 8. File Change Map

- Modify: `RAG Chain.py`
  - Add run context wrapper and stage timers
  - Add structured logger setup
  - Add param/metric/tag/error logging helpers
- Optional new helper module (only if needed for readability): `utils/mlflow_observability.py`
  - Timer utilities, safe logging/truncation helpers, and status logging helper
- Optional new validation input file: `test/observability_validation_questions.json` (20-question set)

### 9. Validation Approach (SC-001 to SC-005)

- Execute 20-question batch via script loop (single process) while MLflow server runs locally.
- Verify SC-001 by counting successful questions vs. created MLflow runs.
- Verify SC-002 by checking trace/span presence per run for all major stages.
- Verify SC-003 by sorting `latency_total_ms` and identifying top 3 slow runs.
- Verify SC-004 by reviewing failed/low-quality runs and confirming retrieval/timing diagnostics identify likely root causes.
- Verify SC-005 via walkthrough task: reviewer inspects runs/traces without opening source and reports comprehension success rate.

## Constitution Check (Post-Design)

Post-Design Gate Status: PASS

- Reproducibility: PASS. Uses explicit experiment assignment and environment-driven tracking setup.
- Dependency Management: PASS. No extra dependencies introduced.
- Testing Discipline: PASS. Quickstart defines local acceptance validation; if helper module added, tests are required.
- Notebook-to-Module Migration: PASS. No notebook duplication introduced.
- Incremental Refactoring: PASS. Observability-only layering with behavior-preserving constraints documented.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
