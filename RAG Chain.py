#!/usr/bin/env python
# coding: utf-8

"""Minimal (non-agent) RAG over a Chroma vector store using LangChain and Ollama.

Required packages:
    pip install -U langchain langchain-chroma langchain-ollama python-dotenv

Assumes the Chroma collection has already been populated (see "Semantic Search.py").

Pipeline (naive RAG — no agent, no tool, single pass):
    1. Set up the Ollama LLM and embeddings.
    2. Connect to the existing Chroma vector store.
    3. Retrieve the top-k chunks for the question.
    4. Stuff them into a prompt as context.
    5. Ask the LLM once and return the answer.
"""

import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Structured logger for observability diagnostics
logger = logging.getLogger("rag_chain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(stage)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# Configuration
LLM_MODEL = "qwen3:0.6b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "pdf_collection"
PERSIST_DIRECTORY = "./chroma_db"
EXPERIMENT_NAME = "minimal-rag-chain"

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say so. Cite page numbers.

Context:
{context}

Question: {question}

Answer:"""
)


def create_llm():
    """Create the Ollama chat LLM."""
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


def create_embeddings():
    """Create the Ollama embedding function."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def connect_vector_store(embeddings):
    """Connect to the existing Chroma vector store."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    count = vector_store._collection.count()
    print(f"✓ Using Ollama {LLM_MODEL} + {EMBEDDING_MODEL}")
    print(f"✓ Connected to existing Chroma vector store ({count} documents)")
    return vector_store


def format_docs(docs):
    """Flatten retrieved documents into a single context string."""
    return "\n\n".join(
        f"Page {doc.metadata.get('page', '?')}: {doc.page_content}" for doc in docs
    )


def build_chain(llm, vector_store, k=4):
    """Wire retriever -> prompt -> llm -> string into one runnable.

    Unlike the agent version, the LLM never decides whether to retrieve:
    every question always triggers exactly one retrieval, then one answer.

    Returns (chain, retriever) so callers can instrument individual stages.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(chain, retriever, llm, question):
    """Ask the RAG chain a question with full observability instrumentation.

    Decomposes the chain into individually traced stages:
    retrieval → prompt_composition → model_invocation → response_formatting.
    Each invocation creates exactly one MLflow run with a linked end-to-end trace.

    Returns the answer string. Prints question/answer to console.
    """
    PREVIEW_MAX = 2000  # Max chars for logged previews (T024)

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    run_name = f"rag_chain__{LLM_MODEL}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    total_start = time.time()
    answer = None
    current_stage = "setup"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(
            "Run started: run_id=%s, run_name=%s",
            run_id, run_name,
            extra={"stage": "run"},
        )

        # Initialize timing accumulators for finally block
        retrieval_ms = prompt_ms = model_ms = formatting_ms = 0.0
        docs = []
        context = ""

        try:
            # Stage 1: Retrieval
            current_stage = "retrieval"
            logger.info("Stage begin", extra={"stage": "retrieval"})
            retrieval_start = time.time()
            with mlflow.start_span("retrieval") as span:
                docs = retriever.invoke(question)
                retrieval_ms = (time.time() - retrieval_start) * 1000
                span.set_attributes({
                    "retrieved_doc_count": len(docs),
                    "latency_ms": retrieval_ms,
                })
            logger.info(
                "Stage end: doc_count=%d, latency_ms=%.1f",
                len(docs), retrieval_ms,
                extra={"stage": "retrieval"},
            )

            # T023: Handle zero-retrieval edge case
            if len(docs) == 0:
                logger.warning(
                    "Zero documents retrieved — chain will proceed with empty context",
                    extra={"stage": "retrieval"},
                )

            # Stage 2: Prompt composition
            current_stage = "prompt_composition"
            logger.info("Stage begin", extra={"stage": "prompt_composition"})
            prompt_start = time.time()
            with mlflow.start_span("prompt_composition") as span:
                context = format_docs(docs)
                prompt_value = RAG_PROMPT.invoke({
                    "context": context,
                    "question": question,
                })
                prompt_ms = (time.time() - prompt_start) * 1000
                span.set_attributes({
                    "context_char_count": len(context),
                    "latency_ms": prompt_ms,
                })
            logger.info(
                "Stage end: context_chars=%d, latency_ms=%.1f",
                len(context), prompt_ms,
                extra={"stage": "prompt_composition"},
            )

            # Stage 3: Model invocation
            current_stage = "model_invocation"
            logger.info("Stage begin", extra={"stage": "model_invocation"})
            model_start = time.time()
            with mlflow.start_span("model_invocation") as span:
                llm_output = llm.invoke(prompt_value)
                model_ms = (time.time() - model_start) * 1000
                span.set_attributes({
                    "model": LLM_MODEL,
                    "latency_ms": model_ms,
                })
            logger.info(
                "Stage end: model=%s, latency_ms=%.1f",
                LLM_MODEL, model_ms,
                extra={"stage": "model_invocation"},
            )

            # Stage 4: Response formatting
            current_stage = "response_formatting"
            logger.info("Stage begin", extra={"stage": "response_formatting"})
            formatting_start = time.time()
            with mlflow.start_span("response_formatting") as span:
                answer = StrOutputParser().invoke(llm_output)
                formatting_ms = (time.time() - formatting_start) * 1000
                span.set_attributes({
                    "answer_char_count": len(answer),
                    "latency_ms": formatting_ms,
                })
            logger.info(
                "Stage end: answer_chars=%d, latency_ms=%.1f",
                len(answer), formatting_ms,
                extra={"stage": "response_formatting"},
            )

            # T021: Success path
            mlflow.set_tag("run_status", "success")
            mlflow.log_metric("error_flag", 0)
            logger.info(
                "Run status: success",
                extra={"stage": "run"},
            )

        except Exception as exc:
            # T022: Failure path — record error details, then re-raise
            total_ms = (time.time() - total_start) * 1000
            mlflow.set_tag("run_status", "failure")
            mlflow.log_metric("error_flag", 1)
            mlflow.set_tag("error_type", type(exc).__name__)
            mlflow.set_tag("error_stage", current_stage)
            mlflow.log_text(traceback.format_exc(), "error_details.txt")
            logger.error(
                "Run status: failure — error_type=%s, error_stage=%s, msg=%s",
                type(exc).__name__, current_stage, str(exc),
                extra={"stage": "run"},
            )
            raise

        finally:
            total_ms = (time.time() - total_start) * 1000

            # --- US2: Static config params (T014) ---
            mlflow.log_params({
                "llm_model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "search_type": "similarity",
                "retriever_k": 4,
                "collection_name": COLLECTION_NAME,
                "persist_directory": PERSIST_DIRECTORY,
            })

            # --- US2: Categorical tags (T015) ---
            question_hash = hashlib.sha256(question.encode()).hexdigest()[:12]
            mlflow.set_tags({
                "component": "rag-chain",
                "observability_version": "v1",
                "tracking_mode": "local",
                "question_hash": question_hash,
            })

            # --- US2: Timing metrics (T016) ---
            mlflow.log_metrics({
                "latency_total_ms": total_ms,
                "latency_retrieval_ms": retrieval_ms,
                "latency_prompt_composition_ms": prompt_ms,
                "latency_model_invocation_ms": model_ms,
                "latency_response_formatting_ms": formatting_ms,
            })

            # --- US2: Retrieval diagnostics (T017) ---
            mlflow.log_metrics({
                "retrieved_doc_count": len(docs),
                "context_char_count": len(context),
            })
            mlflow.set_tag("retrieval_empty", str(len(docs) == 0).lower())

            # --- US2: Answer diagnostics (T018) ---
            if answer is not None:
                mlflow.log_metric("answer_char_count", len(answer))
                # T024: Truncate logged preview if oversized
                answer_preview = answer[:PREVIEW_MAX]
                question_preview = question[:PREVIEW_MAX]
                if len(answer) > PREVIEW_MAX:
                    mlflow.set_tag("answer_preview_truncated", "true")
                if len(question) > PREVIEW_MAX:
                    mlflow.set_tag("question_preview_truncated", "true")
                mlflow.log_text(
                    f"Question: {question_preview}\n\nAnswer: {answer_preview}",
                    "question_answer.txt",
                )

            # --- US2: Run diagnostics artifact (T019) ---
            diagnostics = {
                "run_id": run_id,
                "question": question[:PREVIEW_MAX],
                "question_hash": question_hash,
                "answer_preview": (answer[:500] if answer else None),
                "retrieved_doc_count": len(docs),
                "context_char_count": len(context),
                "answer_char_count": (len(answer) if answer else 0),
                "latency_total_ms": round(total_ms, 1),
                "latency_retrieval_ms": round(retrieval_ms, 1),
                "latency_prompt_composition_ms": round(prompt_ms, 1),
                "latency_model_invocation_ms": round(model_ms, 1),
                "latency_response_formatting_ms": round(formatting_ms, 1),
            }
            mlflow.log_text(
                json.dumps(diagnostics, indent=2), "run_diagnostics.json"
            )

            # --- US2: Structured diagnostic logs (T020) ---
            logger.info(
                "Diagnostics: doc_count=%d, context_chars=%d, answer_chars=%d, "
                "total_ms=%.1f",
                len(docs), len(context),
                (len(answer) if answer else 0), total_ms,
                extra={"stage": "diagnostics"},
            )

            logger.info(
                "Run completed: total_ms=%.1f", total_ms,
                extra={"stage": "run"},
            )

    if answer is not None:
        print(f"\n💬 Answer:\n{answer}")
    return answer


def main():
    """Set up the RAG chain and run a sample query."""
    load_dotenv()

    # T025: Handle unreachable MLflow URI — degrade to console-only logging
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.langchain.autolog()
    except Exception as exc:
        logger.warning(
            "MLflow tracking unavailable (%s: %s) — falling back to console-only",
            type(exc).__name__, exc,
            extra={"stage": "setup"},
        )

    llm = create_llm()
    embeddings = create_embeddings()
    vector_store = connect_vector_store(embeddings)

    chain, retriever = build_chain(llm, vector_store, k=4)

    # T028: Batch validation — run all 20 questions from validation set
    validation_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test",
        "observability_validation_questions.json",
    )
    if os.path.exists(validation_path) and "--batch" in sys.argv:
        with open(validation_path) as f:
            questions = json.load(f)
        print(f"\n🔄 Running batch validation with {len(questions)} questions\n")
        success_count = 0
        fail_count = 0
        for i, q in enumerate(questions, 1):
            try:
                print(f"\n--- Question {i}/{len(questions)} ---")
                ask(chain, retriever, llm, q)
                success_count += 1
            except Exception as exc:
                fail_count += 1
                logger.error(
                    "Batch question %d failed: %s", i, exc,
                    extra={"stage": "batch"},
                )
        print(f"\n✓ Batch complete: {success_count} success, {fail_count} failed")
    else:
        # Single-question demo mode
        ask(chain, retriever, llm, "What are the main methods mentioned in this paper?")


if __name__ == "__main__":
    main()
