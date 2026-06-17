# Research: Add MLflow Observability

## Decision 1: Use explicit run lifecycle with `mlflow.start_run()` per chain invocation

- Decision: Wrap each question invocation in a dedicated `with mlflow.start_run(...)` block while keeping `mlflow.set_experiment(...)` at startup.
- Rationale: This guarantees FR-001 one-run-per-invocation behavior and enables predictable run comparison fields across the 20-question validation set.
- Alternatives considered:
  - Rely only on implicit autolog run creation: rejected because run boundaries can become less explicit in mixed scripted flows.
  - Single long-lived run for all questions: rejected because it fails per-invocation comparability.

## Decision 2: Combine LangChain autolog with manual stage spans

- Decision: Keep `mlflow.langchain.autolog()` and add manual `mlflow.start_span()` boundaries for retrieval, prompt composition, model invocation, and response formatting.
- Rationale: Autolog captures runnable internals, while manual span naming enforces complete, consistent stage coverage for FR-002/FR-007 and SC-002.
- Alternatives considered:
  - Manual spans only: rejected because it misses convenient LangChain-native tracing details already available.
  - Decorator-only tracing (`@mlflow.trace`) without stage spans: rejected because stage-level granularity may be insufficient for SC-003/SC-004 diagnostics.

## Decision 3: Use MLflow params/tags/metrics split with a compact diagnostics artifact

- Decision: Log static config as params, categorical context as tags, numeric observability signals as metrics, and detailed per-run diagnostics in a JSON artifact.
- Rationale: This mirrors MLflow conventions and supports filtering, charting, and side-by-side comparison while keeping UI-friendly display.
- Alternatives considered:
  - Tags only: rejected because numeric analysis (latency ranking) is harder.
  - Artifact-only logging: rejected because query/filter operations in UI become cumbersome.

## Decision 4: Preserve behavior through read-only instrumentation wrappers

- Decision: Layer observability in wrappers that measure and record data without modifying chain inputs/outputs.
- Rationale: Satisfies FR-009 and constitution incremental refactoring constraints.
- Alternatives considered:
  - Rebuild chain into a new abstraction layer: rejected as unnecessary risk for tutorial clarity.

## Decision 5: Handle edge cases with fail-open logging policy

- Decision: If tracking backend is unavailable, continue chain execution with structured console logs; if model/retrieval stages fail, capture status and exception details then re-raise.
- Rationale: Keeps tutorial executable while preserving transparent failure reporting (FR-006, edge cases).
- Alternatives considered:
  - Hard-fail on tracking connectivity issues: rejected because observability should not block core chain behavior.
  - Swallow model exceptions: rejected because it hides runtime issues and harms debugging.

## Decision 6: Structured Python logging with stable key-value context

- Decision: Configure a module-level logger using timestamped, stage-aware key-value messages suitable for terminal walkthroughs.
- Rationale: Meets FR-010 and keeps diagnostics readable in local tutorial sessions.
- Alternatives considered:
  - Verbose raw dumps of full context/answers: rejected due to noise and accidental disclosure risk.
  - No logging beyond MLflow UI: rejected because local live-debug feedback is needed.

## Decision 7: Validation via scripted 20-question batch and MLflow UI checks

- Decision: Add a local validation procedure that runs a 20-question set and verifies SC-001 through SC-005 by run counts, stage coverage, latency ranking, and manual comprehension checks.
- Rationale: Directly maps to spec success criteria and is feasible in a tutorial environment.
- Alternatives considered:
  - Unit-test-only validation: rejected because UI trace inspection and run diagnostics need end-to-end execution.
