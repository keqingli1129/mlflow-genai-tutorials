# Tasks: Add MLflow Observability

**Input**: Design documents from `/specs/001-add-mlflow-observability/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in spec. Tests omitted unless helper module is added to `utils/` (constitution III).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Script-first layout at repository root
- Modified file: `RAG Chain.py`
- Optional helper module: `utils/mlflow_observability.py`
- Validation data: `test/observability_validation_questions.json`

---

## Phase 1: Setup

**Purpose**: Prepare imports, logger, and timing utilities needed by all user stories

- [X] T001 Add `import logging`, `import time`, `import hashlib`, `import json`, `import traceback` to `RAG Chain.py`
- [X] T002 [P] Configure module-level structured Python logger with timestamped stage-aware format in `RAG Chain.py`
- [X] T003 [P] Create 20-question validation data file at `test/observability_validation_questions.json`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core observability infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Refactor `ask()` to separate chain invocation from print output, keeping return value contract unchanged in `RAG Chain.py`
- [X] T005 Refactor `build_chain()` to return both chain and retriever reference so stage-level instrumentation can access retriever independently in `RAG Chain.py`

**Checkpoint**: Foundation ready — `ask()` and `build_chain()` signatures support observability wrapping without changing functional behavior

---

## Phase 3: User Story 1 — End-to-End Run Visibility (Priority: P1) 🎯 MVP

**Goal**: Each chain invocation creates exactly one MLflow run with a linked end-to-end trace covering retrieval, prompt composition, model invocation, and response formatting stages.

**Independent Test**: Run `uv run python "RAG Chain.py"` once, open MLflow UI, confirm one new run exists under experiment `minimal-rag-chain` with a trace containing all four stage spans.

### Implementation for User Story 1

- [X] T006 [US1] Wrap invocation inside `with mlflow.start_run(run_name=...)` in `ask()` using naming convention `rag_chain__{model}__{YYYYMMDD_HHMMSS}` in `RAG Chain.py`
- [X] T007 [US1] Retain `mlflow.langchain.autolog()` call in `main()` and verify it executes before any chain invocation in `RAG Chain.py`
- [X] T008 [US1] Add manual `mlflow.start_span("retrieval")` around the retriever invocation stage in `RAG Chain.py`
- [X] T009 [P] [US1] Add manual `mlflow.start_span("prompt_composition")` around prompt template rendering stage in `RAG Chain.py`
- [X] T010 [P] [US1] Add manual `mlflow.start_span("model_invocation")` around LLM call stage in `RAG Chain.py`
- [X] T011 [P] [US1] Add manual `mlflow.start_span("response_formatting")` around output parsing stage in `RAG Chain.py`
- [X] T012 [US1] Verify trace-to-run linkage by confirming all spans execute within active run context in `RAG Chain.py`
- [X] T013 [US1] Add structured log messages for stage begin/end events with run_id context in `RAG Chain.py`

**Checkpoint**: User Story 1 complete — single-question invocation produces one run with linked 4-stage trace visible in MLflow UI

---

## Phase 4: User Story 2 — Rich Run Diagnostics (Priority: P2)

**Goal**: Each run includes human-reviewable question/answer text, retrieval diagnostics, and timing metrics that enable per-run comparison and slow-run identification.

**Independent Test**: Execute 3+ different questions, open MLflow UI, confirm each run has comparable params/metrics/tags and that the slowest run is identifiable by `latency_total_ms`.

### Implementation for User Story 2

- [X] T014 [US2] Log static config params (`llm_model`, `embedding_model`, `search_type`, `retriever_k`, `collection_name`, `persist_directory`) via `mlflow.log_params()` in `RAG Chain.py`
- [X] T015 [US2] Log categorical tags (`component=rag-chain`, `observability_version=v1`, `tracking_mode=local`, `question_hash`) via `mlflow.set_tags()` in `RAG Chain.py`
- [X] T016 [US2] Log timing metrics (`latency_total_ms`, `latency_retrieval_ms`, `latency_prompt_composition_ms`, `latency_model_invocation_ms`, `latency_response_formatting_ms`) via `mlflow.log_metrics()` in `RAG Chain.py`
- [X] T017 [P] [US2] Log retrieval diagnostics metrics (`retrieved_doc_count`, `context_char_count`) and tag (`retrieval_empty`) in `RAG Chain.py`
- [X] T018 [P] [US2] Log answer diagnostics metric (`answer_char_count`) and log question/answer text via `mlflow.log_input()` / `mlflow.log_text()` in `RAG Chain.py`
- [X] T019 [US2] Generate and log compact JSON diagnostics artifact (`run_diagnostics.json`) per run via `mlflow.log_artifact()` in `RAG Chain.py`
- [X] T020 [US2] Add structured log messages for diagnostic values (doc_count, latency_ms, char counts) at each stage in `RAG Chain.py`

**Checkpoint**: User Stories 1 AND 2 complete — runs have full telemetry for comparison; slowest run identifiable by sorting `latency_total_ms`

---

## Phase 5: User Story 3 — Audit-Ready Logging for Tutorial Iteration (Priority: P3)

**Goal**: Run status is recorded (success/failure) with exception details; edge cases are handled gracefully; structured console logs support offline debugging and regression analysis.

**Independent Test**: Trigger a failure scenario (e.g., wrong model name) and verify the run records `run_status=failure`, `error_flag=1`, error type/stage tags, and an `error_details.txt` artifact. Verify normal runs record `run_status=success`.

### Implementation for User Story 3

- [X] T021 [US3] Wrap chain invocation in `try/except/finally` block; set tag `run_status=success` and metric `error_flag=0` on success path in `RAG Chain.py`
- [X] T022 [US3] On exception: set tag `run_status=failure`, metric `error_flag=1`, tags `error_type` and `error_stage`; log traceback artifact `error_details.txt` via `mlflow.log_text()` then re-raise in `RAG Chain.py`
- [X] T023 [P] [US3] Handle zero-retrieval edge case: log `retrieved_doc_count=0`, set tag `retrieval_empty=true`, continue chain execution in `RAG Chain.py`
- [X] T024 [P] [US3] Handle oversized prompt/output edge case: truncate only logged preview strings, set tag `*_preview_truncated=true`, preserve actual chain data unchanged in `RAG Chain.py`
- [X] T025 [US3] Handle unreachable MLflow URI edge case: catch tracking initialization errors, degrade to console-only logging, continue chain execution in `RAG Chain.py`
- [X] T026 [US3] Add structured log messages for run status transitions (success/failure) and error details in `RAG Chain.py`

**Checkpoint**: All 3 user stories complete — runs record success/failure status, edge cases handled, console logs sufficient for offline audit

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validation, non-interference verification, and documentation alignment

- [X] T027 [P] Verify non-interference guarantee (FR-009): confirm chain answer output is byte-identical with and without observability instrumentation for at least 3 questions in `RAG Chain.py`
- [X] T028 [P] Add batch invocation loop in `main()` to run all 20 validation questions from `test/observability_validation_questions.json` in `RAG Chain.py`
- [X] T029 Run quickstart.md validation procedure end-to-end: verify SC-001 (run count), SC-002 (stage coverage), SC-003 (slow-run ranking), SC-004 (retrieval diagnostics)
- [X] T030 Review and update inline code comments for tutorial readability in `RAG Chain.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — delivers MVP
- **US2 (Phase 4)**: Depends on Foundational (Phase 2); benefits from US1 run context but independently testable
- **US3 (Phase 5)**: Depends on Foundational (Phase 2); benefits from US1+US2 context but independently testable
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) — no dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) — works within US1 run context but can be tested independently with its own `mlflow.start_run()`
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) — wraps the invocation path established in US1/US2 but error handling is independently testable

### Within Each User Story

- Stage span tasks (T008–T011) can run in parallel after T006
- Diagnostic logging tasks (T017, T018) can run in parallel after T014–T016
- Edge case tasks (T023, T024) can run in parallel after T021

### Parallel Opportunities

- Setup tasks T002 and T003 can run in parallel
- US1 span tasks T009, T010, T011 can run in parallel (different stage boundaries)
- US2 diagnostic tasks T017, T018 can run in parallel (different metric groups)
- US3 edge case tasks T023, T024 can run in parallel (independent conditions)
- Polish tasks T027, T028 can run in parallel (independent verifications)

---

## Parallel Example: User Story 1

```
T006 (run wrapper)
 ├── T007 (verify autolog) ──→ T012 (trace linkage)
 ├── T008 (retrieval span) ──→ T012
 ├── T009 (prompt span) [P] ──→ T012
 ├── T010 (model span) [P] ──→ T012
 └── T011 (output span) [P] ──→ T012 ──→ T013 (stage logs)
```

## Implementation Strategy

- **MVP scope**: Complete Phases 1–3 (Setup + Foundational + US1) for minimal viable observability
- **Incremental delivery**: Each phase checkpoint is a working, demonstrable state
- **Single file focus**: All changes in `RAG Chain.py`; no new abstraction layers unless readability demands it
