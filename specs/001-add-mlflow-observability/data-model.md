# Data Model: Add MLflow Observability

## Entity: ObservedChainRun

- Description: One MLflow-backed record per question invocation.
- Fields:
  - run_id (string, required)
  - experiment_name (string, required)
  - run_name (string, required)
  - question_text (string, required)
  - answer_text (string, optional if failure)
  - status (enum: success|failure, required)
  - started_at (datetime, required)
  - ended_at (datetime, required)
  - latency_total_ms (float, required)
  - llm_model (string, required)
  - embedding_model (string, required)
  - retriever_k (int, required)
  - retrieval_doc_count (int, required)
  - context_char_count (int, required)
  - answer_char_count (int, optional)
  - error_type (string, optional)
  - error_message (string, optional)
  - error_stage (string, optional)
- Validation rules:
  - status=success requires answer_text and answer_char_count.
  - status=failure requires error_type and error_stage when exception exists.
  - latency_total_ms >= 0.

## Entity: StageTrace

- Description: Stage-level trace span attached to a run for pipeline observability.
- Fields:
  - span_name (enum: retrieval|prompt_composition|model_invocation|response_formatting, required)
  - run_id (string, required)
  - started_at (datetime, required)
  - ended_at (datetime, required)
  - latency_ms (float, required)
  - stage_status (enum: success|failure, required)
  - stage_attributes (map<string, scalar>, optional)
- Validation rules:
  - Every successful run should contain all 4 required spans.
  - latency_ms >= 0.

## Entity: RetrievalDiagnosticRecord

- Description: Retrieval-specific observability payload for each invocation.
- Fields:
  - run_id (string, required)
  - search_type (string, required)
  - retriever_k (int, required)
  - retrieved_doc_count (int, required)
  - retrieved_pages_preview (list<string>, optional)
  - retrieval_empty (bool, required)
- Validation rules:
  - retrieval_empty=true iff retrieved_doc_count==0.

## Entity: RunComparisonSnapshot

- Description: Normalized subset of run fields used for ranking and comparison.
- Fields:
  - run_id (string, required)
  - question_hash (string, required)
  - status (enum, required)
  - latency_total_ms (float, required)
  - retrieval_doc_count (int, required)
  - context_char_count (int, required)
  - answer_char_count (int, optional)
- Validation rules:
  - Fields must be logged consistently for every invocation to support SC-003/SC-004 analysis.

## State Transitions

- ObservedChainRun.status:
  - running -> success (when final answer produced and logged)
  - running -> failure (when exception is raised in any stage)
- StageTrace.stage_status:
  - running -> success
  - running -> failure
