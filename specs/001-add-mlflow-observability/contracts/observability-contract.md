# Observability Contract: Minimal RAG Chain

## Purpose

Define the observable interface and required telemetry fields for each chain invocation.

## Invocation Contract

- Entrypoint: `ask(chain, question: str) -> str`
- Functional behavior: unchanged; returns final answer string or raises exception on failure.
- Observability side effects:
  - exactly one MLflow run per invocation
  - stage-level spans
  - required params/metrics/tags/artifacts

## Required Params

- `llm_model` (string)
- `embedding_model` (string)
- `search_type` (string)
- `retriever_k` (int)
- `collection_name` (string)
- `persist_directory` (string)

## Required Metrics

- `latency_total_ms` (float)
- `latency_retrieval_ms` (float)
- `latency_prompt_composition_ms` (float)
- `latency_model_invocation_ms` (float)
- `latency_response_formatting_ms` (float)
- `retrieved_doc_count` (int)
- `context_char_count` (int)
- `answer_char_count` (int, success only)
- `error_flag` (0|1)

## Required Tags

- `component=rag-chain`
- `run_status=success|failure`
- `observability_version=v1`
- `question_hash=<hash>`
- `error_type` (failure only)
- `error_stage` (failure only)
- `retrieval_empty=true|false`

## Required Trace Spans

- `retrieval`
- `prompt_composition`
- `model_invocation`
- `response_formatting`

Each span must include:
- `latency_ms`
- `stage_status`

## Error Semantics

- Exceptions during chain invocation are re-raised unchanged after observability logging.
- Failure run must include `run_status=failure`, `error_flag=1`, and an exception details artifact.

## Non-Interference Clause

Observability code must not mutate:
- question text passed to retriever/LLM
- retrieved document set used for prompt
- prompt template inputs used by answer generation
- returned answer text

Any truncation is allowed only for logging previews and must be tagged as truncated.
