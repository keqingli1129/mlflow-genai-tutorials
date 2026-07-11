import hashlib
import os
import time
from datetime import datetime
from typing import Dict, List

import mlflow
import numpy as np
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)

EXPERIMENT_NAME = "11-complete-rag-system"

# Shared OpenAI/AI Gateway client, populated by setup()
client = None

# Sample document corpus (in production, load from database)
DOCUMENT_STORE = {
    "doc1": "MLflow is an open source developer platform to build AI applications and models with confidence.",
    "doc2": "MLflow Tracing provides comprehensive observability for GenAI applications. It captures LLM calls, retrieval steps, tool usage, and agent reasoning with full input/output visibility.",
    "doc3": "MLflow integrates with 40+ frameworks including OpenAI, Anthropic, LangChain, LlamaIndex, DSPy, and AutoGen. Each integration provides automatic tracing and experiment tracking.",
    "doc4": "MLflow Evaluation enables systematic testing of GenAI applications using LLM-as-judge metrics, custom scorers, and human feedback. It supports both batch and online evaluation.",
    "doc5": "MLflow Prompt Registry allows teams to version, share, and manage prompts centrally. It tracks which prompts are used in which experiments and enables A/B testing.",
    "doc6": "MLflow supports collaborative development with experiment sharing, model versioning, and deployment tracking. Teams can compare results and iterate systematically.",
    "doc7": "MLflow provides cost tracking for LLM applications by monitoring token usage, API calls, and compute resources. This helps teams optimize spending and budget effectively.",
    "doc8": "MLflow is fully open source and vendor-neutral, ensuring no lock-in. It works with any cloud provider, ML framework, or LLM provider.",
    "doc9": "MLflow is a platform for the complete machine learning lifecycle. It provides experiment tracking, model packaging, and deployment capabilities across various ML frameworks.",
    "doc10": "MLflow offers Helpful Assitant, performaance metrics dashboards, Judge Builder online and continous monitoring, and MemAling for dyanmic optization, and more.",
}

# Memory Cache for embeddings but in production, use Redis or something similar vector database
EMBEDDING_CACHE = {}

# Pre-computed document embeddings, populated by compute_document_embeddings()
DOC_EMBEDDINGS = {}

# Test queries
TEST_QUERIES = [
    "What tracing capabilities does MLflow provide?",
    "How does MLflow help with cost tracking?",
    "Can MLflow integrate with LangChain?",
    "What is the purpose of MLflow Prompt Registry?",
]


def setup():
    global client

    # Load environment
    load_dotenv()

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Check if we are using a Databricks AI Gateway client
    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        models = get_ai_gateway_model_names()
        JUDGE_MODEL = models[2]
        AGENT_MODEL = models[0]
        JUDGE_MODEL_URI = f"databricks:/{JUDGE_MODEL}"
    else:
        # Initialize as an OpenAI client
        client = get_openai_client()
        JUDGE_MODEL = "gpt-5.2"
        AGENT_MODEL = "gpt-5.2"
        JUDGE_MODEL_URI = f"openai:/{JUDGE_MODEL}"

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Enable autologging
    mlflow.openai.autolog()

    print("✅ Environment configured for production RAG system")
    print(f"   MLflow tracking: {mlflow.get_tracking_uri()}")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print(f"   Model: {AGENT_MODEL}")
    print(f"   Judge Model: {JUDGE_MODEL}")
    print(f"   Use Databricks Provider: {use_databricks_provider}")

    return JUDGE_MODEL, AGENT_MODEL, JUDGE_MODEL_URI, use_databricks_provider


# creating a span for the embedding function
@mlflow.trace(name="embed_text", span_type="EMBEDDING")
def embed_text(text: str) -> List[float]:
    """
    Generate embeddings with caching.
    """
    # Check cache
    cache_key = hashlib.md5(text.encode()).hexdigest()
    span = mlflow.get_current_active_span()
    if cache_key in EMBEDDING_CACHE:
        span.set_attributes({"cache_hit": True})
        return EMBEDDING_CACHE[cache_key]

    span.set_attributes({"cache_hit": False, "text_length": len(text)})

    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding
    span.set_attributes({"embedding_dim": len(embedding)})

    # Cache for future use
    EMBEDDING_CACHE[cache_key] = embedding

    return embedding


def compute_document_embeddings():
    print(f"📚 Document store initialized with {len(DOCUMENT_STORE)} documents")
    print("✅ Embedding function defined with caching")

    # Pre-compute document embeddings
    print("\n🔄 Computing document embeddings...")

    for doc_id, text in DOCUMENT_STORE.items():
        DOC_EMBEDDINGS[doc_id] = embed_text(text)
        print(f"  ✓ {doc_id}")

    print(f"\n✅ {len(DOC_EMBEDDINGS)} documents embedded and cached")


@mlflow.trace(name="validate_query", span_type="PARSER")
def validate_query(query: str) -> Dict:
    """
    Validate and preprocess user query.
    """
    span = mlflow.get_current_active_span()
    span.set_attributes({"original_length": len(query)})

    # Basic validation
    if not query or len(query.strip()) == 0:
        mlflow.log_span_attribute("validation_error", "empty_query")
        raise ValueError("Query cannot be empty")

    if len(query) > 1000:
        span.set_attributes({"validation_warning": "query_too_long"})
        query = query[:1000]  # Truncate

    # Preprocess
    processed_query = query.strip()
    span.set_attributes({
        "processed_length": len(processed_query),
        "validation_passed": True
    })

    return {
        "original": query,
        "processed": processed_query,
        "valid": True
    }


@mlflow.trace(name="semantic_search", span_type="RETRIEVER")
def search_documents(
    query_embedding: List[float],
    doc_embeddings: Dict[str, List[float]],
    top_k: int = 3,
    min_score: float = 0.7
) -> List[Dict]:
    """
    Search for most relevant documents using cosine similarity.
    """
    span = mlflow.get_current_active_span()
    span.set_attributes({
        "corpus_size": len(doc_embeddings),
        "top_k": top_k,
        "min_score": min_score
    })

    # Calculate cosine similarity for all documents
    scores = {}
    for doc_id, doc_emb in doc_embeddings.items():
        similarity = np.dot(query_embedding, doc_emb) / \
                     (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        scores[doc_id] = float(similarity)

    # Sort by score and filter by minimum threshold
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    filtered_docs = [(doc_id, score) for doc_id, score in sorted_docs if score >= min_score]

    # Get top k
    top_docs = filtered_docs[:top_k]

    # Return in LangChain Document format (page_content + metadata)
    # MLflow's RAGAS integration expects this format in RETRIEVER span outputs
    results = [
        {
            "page_content": DOCUMENT_STORE[doc_id],
            "metadata": {"doc_id": doc_id, "score": score}
        }
        for doc_id, score in top_docs
    ]

    # Log metrics
    if results:
        span.set_attributes({
            "num_results": len(results),
            "top_score": results[0]["metadata"]["score"],
            "avg_score": np.mean([r["metadata"]["score"] for r in results]),
            "min_result_score": results[-1]["metadata"]["score"]
        })
    else:
        span.set_attributes({"num_results": 0,
                             "retrieval_warning": "no_docs_above_threshold"})

    return results


@mlflow.trace(name="assemble_context", span_type="PARSER")
def assemble_context(query: str, retrieved_docs: List[Dict]) -> str:
    """
    Assemble context from retrieved documents and construct prompt.
    """
    span = mlflow.get_current_active_span()
    span.set_attributes({"num_docs": len(retrieved_docs)})

    if not retrieved_docs:
        span.set_attributes({"context_warning": "no_docs_retrieved"})
        return None

    # Format context
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[Document {i}] (Relevance: {doc['metadata']['score']:.2f})\n{doc['page_content']}"
        )

    context = "\n\n".join(context_parts)

    # Construct prompt with system instructions
    prompt = f"""You are a helpful AI assistant that answers questions based on provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below
- If the answer is not in the context, say "I don't have enough information to answer that"
- Be concise but complete
- Cite which document(s) you used if relevant

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    span.set_attributes({
        "context_length": len(context),
        "prompt_length": len(prompt)
    })

    return prompt


@mlflow.trace(name="generate_response", span_type="LLM")
def generate_answer(
    prompt: str,
    model: str = "gpt-5.2",
    temperature: float = 0.1,
) -> Dict:
    """
    Generate answer using LLM.
    """
    span = mlflow.get_current_active_span()
    span.set_attributes({
        "model": model,
        "temperature": temperature
    })

    if not prompt:
        span.set_attributes({"generation_error": "empty_prompt"})
        raise ValueError("Prompt cannot be empty")

    # Call LLM (automatically traced by OpenAI autolog)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    answer = response.choices[0].message.content

    # Log generation metrics
    span.set_attributes({
        "answer_length": len(answer),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason
    })
    span.set_attributes({
        "finish_reason": response.choices[0].finish_reason
    })

    return {
        "answer": answer,
        "tokens": response.usage.total_tokens,
        "model": model,
        "finish_reason": response.choices[0].finish_reason
    }


@mlflow.trace(name="validate_response", span_type="PARSER")
def validate_response(answer: str, min_length: int = 10) -> Dict:
    """
    Validate and post-process generated response.
    """
    span = mlflow.get_current_active_span()
    span.set_attributes({
        "min_length_threshold": min_length,
        "answer_length": len(answer)
    })

    issues = []

    # Check length
    if len(answer) < min_length:
        issues.append("too_short")

    # Check for common error patterns
    error_patterns = [
        "I don't have enough information",
        "I cannot answer",
        "I apologize"
    ]

    for pattern in error_patterns:
        if pattern.lower() in answer.lower():
            issues.append(f"contains_{pattern.replace(' ', '_').lower()}")

    span.set_attributes({
        "validation_issues": ",".join(issues) if issues else "none",
        "is_valid": len(issues) == 0
    })

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "answer": answer
    }


@mlflow.trace(name="rag_pipeline", span_type="CHAIN")
def rag_qa_system(
    user_query: str,
    top_k: int = 3,
    min_score: float = 0.7,
    model: str = "gpt-5.2",
    temperature: float = 0.1
) -> Dict:
    """
    Complete RAG pipeline with full observability.

    Returns:
        Dict with answer, metadata, and status
    """
    # Start MLflow run for tracking
    with mlflow.start_run(run_name="rag_query"):

        # Log parameters
        mlflow.log_params({
            "query": user_query[:100],  # Truncate for readability
            "top_k": top_k,
            "min_score": min_score,
            "model": model,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Step 1: Validate query
            validation_result = validate_query(user_query)
            processed_query = validation_result["processed"]

            # Step 2: Generate query embedding
            query_embedding = embed_text(processed_query)

            # Step 3: Search documents
            retrieved_docs = search_documents(
                query_embedding,
                DOC_EMBEDDINGS,
                top_k=top_k,
                min_score=min_score
            )

            if not retrieved_docs:
                mlflow.log_metric("retrieval_success", 0)
                return {
                    "status": "no_relevant_docs",
                    "answer": "I couldn't find relevant information to answer your question.",
                    "query": user_query,
                    "retrieved_docs": []
                }

            mlflow.log_metric("num_docs_retrieved", len(retrieved_docs))
            mlflow.log_metric("avg_relevance_score",
                            np.mean([d["metadata"]["score"] for d in retrieved_docs]))
            mlflow.log_metric("retrieval_success", 1)

            # Step 4: Assemble context and construct prompt
            prompt = assemble_context(processed_query, retrieved_docs)

            # Step 5: Generate answer
            generation_result = generate_answer(
                prompt,
                model=model,
                temperature=temperature
            )

            # Step 6: Validate response
            validation = validate_response(generation_result["answer"])

            # Log final metrics
            mlflow.log_metric("total_tokens", generation_result["tokens"])
            mlflow.log_metric("answer_valid", 1 if validation["is_valid"] else 0)
            mlflow.log_metric("answer_length", len(generation_result["answer"]))

            # Log artifacts
            mlflow.log_text(user_query, "query.txt")
            mlflow.log_text(generation_result["answer"], "answer.txt")
            mlflow.log_text(prompt, "full_prompt.txt")
            mlflow.log_dict(
                {"docs": retrieved_docs},
                "retrieved_docs.json"
            )

            # Construct result
            result = {
                "status": "success",
                "query": user_query,
                "answer": generation_result["answer"],
                "retrieved_docs": retrieved_docs,
                "metadata": {
                    "num_docs": len(retrieved_docs),
                    "avg_relevance": float(np.mean([d["metadata"]["score"] for d in retrieved_docs])),
                    "tokens_used": generation_result["tokens"],
                    "model": model,
                    "is_valid": validation["is_valid"],
                    "validation_issues": validation["issues"]
                }
            }

            return result

        except Exception as e:
            # Log error
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.log_param("error_message", str(e))
            mlflow.log_metric("pipeline_success", 0)

            return {
                "status": "error",
                "query": user_query,
                "error": str(e),
                "error_type": type(e).__name__
            }


def confirm_pipeline_defined():
    print("✅ Query validation function defined")
    print("✅ Semantic search function defined")
    print("✅ Context assembly function defined")
    print("✅ Answer generation function defined")
    print("✅ Response validation function defined")
    print("✅ Complete RAG pipeline defined")


def test_rag_system():
    print("\n🧪 Testing RAG System\n")
    print("=" * 80)

    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        start_time = time.time()
        result = rag_qa_system(query, top_k=3, min_score=0.7)
        latency = time.time() - start_time

        if result["status"] == "success":
            print(f"\nAnswer: {result['answer']}")
            print("\nMetadata:")
            print(f"  - Documents used: {result['metadata']['num_docs']}")
            print(f"  - Avg relevance: {result['metadata']['avg_relevance']:.3f}")
            print(f"  - Tokens: {result['metadata']['tokens_used']}")
            print(f"  - Latency: {latency:.2f}s")
            print(f"  - Valid: {result['metadata']['is_valid']}")

            results.append({
                "query": query,
                "success": True,
                "latency": latency,
                "tokens": result['metadata']['tokens_used']
            })
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
            results.append({
                "query": query,
                "success": False,
                "latency": latency
            })

    print("\n" + "=" * 80)
    print("\n✅ All queries processed!")

    return results


def performance_analysis(results):
    # Analyze performance
    print("\n📊 Performance Summary\n")
    print("=" * 60)

    successful = [r for r in results if r["success"]]

    if successful:
        latencies = [r["latency"] for r in successful]
        tokens = [r["tokens"] for r in successful]

        print(f"Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print("\nLatency Stats:")
        print(f"  Average: {np.mean(latencies):.2f}s")
        print(f"  Min: {np.min(latencies):.2f}s")
        print(f"  Max: {np.max(latencies):.2f}s")
        print(f"  Std Dev: {np.std(latencies):.2f}s")

        print("\nToken Usage:")
        print(f"  Average: {np.mean(tokens):.0f} tokens")
        print(f"  Total: {np.sum(tokens):.0f} tokens")
        print(f"  Est. Cost: ${np.sum(tokens) * 0.15 / 1_000_000:.6f}")

        print("\nCache Performance:")
        print(f"  Embedding cache hits: {len(EMBEDDING_CACHE)} embeddings cached")

    print("\n" + "=" * 60)


def configure_async_compat():
    # Workaround for async event loop issues in Jupyter notebooks
    # Libraries like RAGAS run async code that can conflict with Jupyter's event loop

    import logging

    # 1. Use nest_asyncio to allow nested event loops (essential for Jupyter + async libraries)
    import nest_asyncio
    nest_asyncio.apply()

    # 2. Suppress noisy asyncio error messages (they don't affect results)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # 3. If litellm is installed, disable its async logging to prevent conflicts
    try:
        import litellm
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        litellm.disable_streaming_logging = True
        litellm.turn_off_message_logging = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        print("   - LiteLLM async logging disabled")
    except ImportError:
        pass  # litellm not installed, no workaround needed

    print("✅ Jupyter async compatibility configured")
    print("   - nest_asyncio applied for nested event loop support")
    print("   - Asyncio error logging suppressed")


def initialize_ragas_scorers(judge_model_uri):
    from mlflow.genai.scorers.ragas import ContextRelevance, Faithfulness

    # Initialize RAGAS scorers (requires: pip install ragas)
    # Note: We're using Faithfulness and ContextRelevance which work with traces containing RETRIEVER spans
    # ContextPrecision is not included because it requires expectations['expected_output']

    faithfulness_scorer = Faithfulness(model=judge_model_uri)
    context_relevance_scorer = ContextRelevance(model=judge_model_uri)

    print("✅ RAGAS scorers initialized:")
    print("   - Faithfulness (checks if answer is grounded in retrieved context)")
    print("   - ContextRelevance (checks if retrieved context is relevant to query)")
    print("\n⚠️  ContextPrecision not used - it requires expected_output in expectations")

    return faithfulness_scorer, context_relevance_scorer


def get_rag_traces():
    # Get traces from the RAG pipeline runs (which contain RETRIEVER spans)
    # RAGAS scorers like Faithfulness and ContextRelevance extract context
    # from RETRIEVER spans in traces, not from static datasets

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    # Search for traces from our RAG pipeline, limited to the most recent batch
    # (avoids picking up old traces that may have different output formats)
    rag_traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        filter_string="name = 'rag_pipeline'",
        max_results=len(TEST_QUERIES),
    )

    print(f"✅ Found {len(rag_traces)} RAG pipeline traces for evaluation")
    print("\nThese traces contain RETRIEVER spans that RAGAS scorers need:")
    print("   - semantic_search span with retrieved documents")
    print("   - Full input/output flow for faithfulness checking")

    # Verify traces have the required page_content field in RETRIEVER outputs
    from mlflow.entities import Trace as TraceEntity
    sample_trace = rag_traces.iloc[0]['trace']
    if isinstance(sample_trace, str):
        sample_trace = TraceEntity.from_json(sample_trace)
    retriever_spans = [s for s in sample_trace.data.spans if s.span_type == 'RETRIEVER']
    if retriever_spans:
        outputs = retriever_spans[0].outputs
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
            has_page_content = 'page_content' in outputs[0]
            print(f"\n   Retriever output keys: {list(outputs[0].keys())}")
            print(f"   Has page_content: {has_page_content}")
            if not has_page_content:
                print("\n⚠️  WARNING: Traces missing 'page_content' in RETRIEVER outputs!")
                print("   Re-run from Step 5 to regenerate traces with the correct format.")

    return rag_traces


def run_ragas_evaluation(rag_traces, faithfulness_scorer, context_relevance_scorer):
    # Run RAGAS evaluation on traces (which contain RETRIEVER spans)
    # Note: ContextPrecision requires expected_output in expectations, so we skip it here
    # To use ContextPrecision, you'd need to log expectations to traces:
    #   mlflow.log_expectation(trace_id, name='expected_output', value='...', source='...')

    print("🔄 Running RAGAS evaluation on RAG traces...\n")

    ragas_results = mlflow.genai.evaluate(
        data=rag_traces,  # Pass traces (not static data) - they contain RETRIEVER spans
        scorers=[
            faithfulness_scorer,      # Checks if answer is grounded in retrieved context
            context_relevance_scorer, # Checks if retrieved context is relevant to query
        ]
    )

    print("\n✅ RAGAS evaluation complete!")
    print("\n📊 RAGAS Metrics Summary:")
    print("-" * 50)
    for metric_name, value in ragas_results.metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")


def ui_analysis_guide():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Analyzing Results in MLflow UI                       ║
╚══════════════════════════════════════════════════════════════╝

🔍 EXPERIMENTS VIEW:
   Navigate to: http://localhost:5000
   Select: "10-complete-rag-system" experiment

   You'll see:
   - All RAG query runs
   - Parameters (query, model, top_k)
   - Metrics (tokens, relevance, latency)
   - Artifacts (queries, answers, prompts)

📊 COMPARING RUNS:
   1. Select multiple runs
   2. Click "Compare"
   3. View side-by-side:
      - Which queries used most tokens?
      - Which had highest relevance scores?
      - Performance variations

🌳 TRACES VIEW:
   Click "Traces" tab to see:

   Timeline visualization:
   rag_pipeline (CHAIN) ━━━━━━━━━━━━━━━━━━━━ 2.5s
   ├─ validate_query (PARSER) ━━ 0.01s
   ├─ embed_text (EMBEDDING) ━━━━ 0.3s
   ├─ semantic_search (RETRIEVER) ━ 0.05s
   ├─ assemble_context (PARSER) ━ 0.02s
   ├─ generate_response (LLM) ━━━━━ 2.0s
   │  └─ OpenAI API call ━━━━━━━━━ 1.9s
   └─ validate_response (PARSER) ━ 0.01s

🔎 SPAN DETAILS:
   Click on any span to see:
   - Inputs and outputs
   - Custom attributes
   - Timing information
   - Cache hit status
   - Relevance scores

📈 KEY INSIGHTS:
   1. Performance Bottlenecks:
      - Which step takes longest?
      - Is it the LLM or retrieval?

   2. Quality Metrics:
      - Average relevance scores
      - Documents per query
      - Answer validation rates

   3. Cost Analysis:
      - Token usage per query
      - Cache effectiveness
      - Cost per operation

   4. Error Patterns:
      - Failed queries
      - Low relevance scores
      - Validation issues

💡 OPTIMIZATION OPPORTUNITIES:
   Based on traces, you can:
   - Adjust top_k if retrieval is slow
   - Increase min_score if quality is poor
   - Optimize prompts to reduce tokens
   - Add more aggressive caching
   - Implement parallel retrieval

""")


def main():
    judge_model, agent_model, judge_model_uri, use_databricks_provider = setup()

    compute_document_embeddings()
    confirm_pipeline_defined()

    results = test_rag_system()
    performance_analysis(results)

    configure_async_compat()
    faithfulness_scorer, context_relevance_scorer = initialize_ragas_scorers(judge_model_uri)
    rag_traces = get_rag_traces()
    run_ragas_evaluation(rag_traces, faithfulness_scorer, context_relevance_scorer)

    ui_analysis_guide()


if __name__ == "__main__":
    main()
