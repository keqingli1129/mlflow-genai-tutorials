import hashlib
import json
import os
import sqlite3
from collections import Counter
from typing import Dict, List, Optional, TypedDict

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from mlflow.entities import Feedback
from mlflow.genai import scorer

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_databricks_ai_gateway_langchain_client,
    get_databricks_langchain_chat_client,
    get_langchain_chat_openai_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
    is_databricks_client,
)

# Shared runtime handles populated by setup() / the loader functions and reused
# by the module-level traced functions and graph nodes below.
client = None
llm = None
embed_model_name = None
db_conn = None
POLICY_DOCUMENTS = {}
DOC_EMBEDDINGS = {}
EMBEDDING_CACHE = {}
supervisor = None


def setup():
    load_dotenv()

    print(f"MLflow version: {mlflow.__version__}")

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    global client, llm, embed_model_name

    # Determine provider and initialize clients
    use_databricks_provider = is_databricks_client()
    use_databricks_ai_gateway = is_databricks_ai_gateway_client()

    if use_databricks_ai_gateway:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
        embed_model_name = get_ai_gateway_model_names()[1] if len(get_ai_gateway_model_names()) > 1 else "text-embedding-3-small"
        llm = get_databricks_ai_gateway_langchain_client(model_name, temperature=0.0)
        provider_name = "Databricks AI Gateway"
    elif use_databricks_provider:
        client = get_openai_client()
        model_name = "gpt-4o"
        embed_model_name = "text-embedding-3-small"
        llm = get_databricks_langchain_chat_client(model_name, temperature=0.0)
        provider_name = "Databricks Workspace"
    else:
        client = get_openai_client()
        model_name = "gpt-5-mini"
        embed_model_name = "text-embedding-3-small"
        llm = get_langchain_chat_openai_client(model_name, temperature=0.0)
        provider_name = "OpenAI"

    if not use_databricks_provider and not use_databricks_ai_gateway and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Enable LangGraph autologging (covers LangGraph via langchain integration)
    mlflow.langchain.autolog()

    EXPERIMENT_NAME = "10-Multi-Agent-Supervisor"
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print(f"Embedding model: {embed_model_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("LangChain autologging: ENABLED")

    return model_name, use_databricks_provider


def load_disaster_data():
    from utils.fema_data import get_disaster_data

    global db_conn

    # Load the FEMA disaster database (200 fabricated records, 2020-2025)
    disaster_data = get_disaster_data()

    print(f"FEMA Disaster Database: {len(disaster_data)} records")
    print(f"Columns: {list(disaster_data.columns)}")
    print(f"Years: {sorted(disaster_data['year'].unique())}")
    print(f"States: {disaster_data['state'].nunique()} unique states")
    print(f"Disaster types: {sorted(disaster_data['disaster_type'].unique())}")

    # Register DataFrame as a SQLite in-memory table for text-to-SQL queries
    db_conn = sqlite3.connect(":memory:")
    disaster_data.to_sql("disaster_data", db_conn, index=False, if_exists="replace")
    print(f"\nSQLite table 'disaster_data' registered ({len(disaster_data)} rows)")


@mlflow.trace(name="execute_data_query", span_type="TOOL")
def execute_data_query(query: str) -> str:
    """
    Genie Agent: Translates natural language to SQL and executes via SQLite.
    In production Databricks, this would be a Genie Space generating SQL
    against Unity Catalog tables. Here we simulate it with SQLite text-to-SQL.
    """
    schema_info = (
        "Table 'disaster_data' columns:\n"
        "  - disaster_id: TEXT (e.g., 'DR-4001')\n"
        "  - year: INTEGER (2020-2025)\n"
        "  - state: TEXT (e.g., 'California', 'Florida')\n"
        "  - disaster_type: TEXT ('Wildfire', 'Hurricane', 'Flood', 'Earthquake', 'Tornado')\n"
        "  - severity: INTEGER (2-5)\n"
        "  - affected_population: INTEGER\n"
        "  - federal_aid_amount: INTEGER (in dollars)\n"
        "  - declaration_date: TEXT (YYYY-MM-DD format)"
    )

    sql_prompt = ChatPromptTemplate.from_template(
        "You are a SQL query translator. Given a natural language query about FEMA disaster data, "
        "produce a SQLite-compatible SQL SELECT statement.\n\n"
        "{schema}\n\n"
        "Respond with ONLY a valid SQL SELECT statement. No markdown, no explanation.\n\n"
        "Query: {query}"
    )

    chain = sql_prompt | llm | StrOutputParser()
    raw_sql = chain.invoke({"schema": schema_info, "query": query})

    # Strip markdown code fences if present
    sql = raw_sql.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1]
        sql = sql.rsplit("```", 1)[0].strip()

    # Execute via SQLite
    try:
        result_df = pd.read_sql_query(sql, db_conn)
    except Exception as e:
        return f"SQL execution error: {e}\n\nGenerated SQL: {sql}"

    # Set trace attributes
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "generated_sql": sql,
            "result_rows": len(result_df),
        })

    return f"SQL: {sql}\n\nResults ({len(result_df)} rows):\n{result_df.to_string(index=False)}"


def test_genie_agent():
    # Quick test of the Genie agent
    print("Test: How many disasters hit California?\n")
    result = execute_data_query("How many disasters hit California in 2024?")
    print(result)


def load_policy_documents():
    from utils.policy_docs import get_policy_documents

    global POLICY_DOCUMENTS
    POLICY_DOCUMENTS = get_policy_documents()

    print(f"Knowledge Base: {len(POLICY_DOCUMENTS)} policy documents")
    for doc_id in POLICY_DOCUMENTS:
        print(f"  - {doc_id} ({len(POLICY_DOCUMENTS[doc_id].split())} words)")


@mlflow.trace(name="embed_text", span_type="EMBEDDING")
def embed_text(text: str) -> List[float]:
    """Generate embeddings with caching."""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    span = mlflow.get_current_active_span()

    if cache_key in EMBEDDING_CACHE:
        if span:
            span.set_attributes({"cache_hit": True})
        return EMBEDDING_CACHE[cache_key]

    if span:
        span.set_attributes({"cache_hit": False, "text_length": len(text)})

    response = client.embeddings.create(
        model=embed_model_name,
        input=text
    )
    embedding = response.data[0].embedding
    EMBEDDING_CACHE[cache_key] = embedding
    return embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_documents():
    # Pre-compute document embeddings
    print("Embedding policy documents...")
    for doc_id, text in POLICY_DOCUMENTS.items():
        DOC_EMBEDDINGS[doc_id] = embed_text(text)
        print(f"  Embedded: {doc_id}")

    print(f"\n{len(DOC_EMBEDDINGS)} documents embedded and cached")


@mlflow.trace(name="search_policy_documents", span_type="RETRIEVER")
def search_policy_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Semantic search over FEMA policy documents."""
    query_embedding = embed_text(query)

    similarities = []
    for doc_id, doc_embedding in DOC_EMBEDDINGS.items():
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append({"doc_id": doc_id, "score": score, "text": POLICY_DOCUMENTS[doc_id]})

    similarities.sort(key=lambda x: x["score"], reverse=True)
    results = similarities[:top_k]

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "top_k": top_k,
            "top_score": results[0]["score"] if results else 0,
            "retrieved_docs": [r["doc_id"] for r in results],
        })

    return results


@mlflow.trace(name="answer_from_documents", span_type="TOOL")
def answer_from_documents(query: str) -> str:
    """Knowledge Assistant: retrieves relevant documents and generates an answer."""
    retrieved = search_policy_documents(query, top_k=3)

    context = "\n\n".join(
        f"[{r['doc_id']}] (relevance: {r['score']:.3f})\n{r['text']}"
        for r in retrieved
    )

    answer_prompt = ChatPromptTemplate.from_template(
        "You are a FEMA policy expert. Answer the question using ONLY the provided documents. "
        "Cite the document names in your answer.\n\n"
        "Documents:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    chain = answer_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "query": query})


def test_knowledge_assistant():
    # Quick test of the Knowledge Assistant
    print("Test: What are FEMA's evacuation protocols?\n")
    result = answer_from_documents("What are FEMA's evacuation protocols for wildfire zones?")
    print(result)


# Define the shared state for the supervisor graph
class SupervisorState(TypedDict):
    query: str                          # User's original query
    route: Optional[str]                # "genie" | "knowledge_assistant" | "both"
    routing_reasoning: Optional[str]    # Why the supervisor chose this route
    genie_response: Optional[str]       # Response from the data agent
    ka_response: Optional[str]          # Response from the document agent
    final_response: Optional[str]       # Synthesized final answer


# Node 1: Supervisor Router -- classifies the query and decides routing
def route_query(state: SupervisorState) -> dict:
    """Supervisor node: classifies the query and decides which subagent(s) to invoke."""
    prompt = ChatPromptTemplate.from_template(
        "You are a supervisor agent that routes queries to specialized subagents.\n\n"
        "Available subagents:\n"
        "- GENIE: Handles structured data queries -- statistics, counts, comparisons, trends, "
        "rankings about FEMA disaster records (years, states, types, severity, population, aid amounts).\n"
        "- KNOWLEDGE_ASSISTANT: Handles policy and procedure questions -- evacuation protocols, "
        "safety guidelines, eligibility criteria, declaration processes, response procedures.\n"
        "- BOTH: When the query needs BOTH data AND policy context to answer fully.\n\n"
        "Query: {query}\n\n"
        'Respond with ONLY valid JSON: {{"route": "genie" or "knowledge_assistant" or "both", "reasoning": "one sentence"}}'
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": state["query"]})

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract route from text
        lower = raw.lower()
        if "both" in lower:
            result = {"route": "both", "reasoning": "Fallback parse"}
        elif "genie" in lower:
            result = {"route": "genie", "reasoning": "Fallback parse"}
        else:
            result = {"route": "knowledge_assistant", "reasoning": "Fallback parse"}

    return {
        "route": result["route"],
        "routing_reasoning": result.get("reasoning", "")
    }


# Node 2: Agent Dispatcher -- calls the appropriate subagent(s)
def execute_agents(state: SupervisorState) -> dict:
    """Dispatch node: invokes the appropriate subagent(s) based on the route."""
    updates = {}
    route = state.get("route", "knowledge_assistant")

    if route in ("genie", "both"):
        updates["genie_response"] = execute_data_query(state["query"])

    if route in ("knowledge_assistant", "both"):
        updates["ka_response"] = answer_from_documents(state["query"])

    return updates


# Node 3: Synthesizer -- combines subagent responses into a unified answer
def synthesize_response(state: SupervisorState) -> dict:
    """Synthesizer node: combines subagent outputs into a coherent final response."""
    parts = []
    if state.get("genie_response"):
        parts.append(f"DATA FINDINGS:\n{state['genie_response']}")
    if state.get("ka_response"):
        parts.append(f"POLICY GUIDANCE:\n{state['ka_response']}")

    combined_context = "\n\n".join(parts)

    synth_prompt = ChatPromptTemplate.from_template(
        "You are a FEMA response coordinator. Synthesize the following information from "
        "specialized agents into a clear, unified response for the user.\n\n"
        "Original question: {query}\n\n"
        "Agent outputs:\n{context}\n\n"
        "Provide a comprehensive answer that:\n"
        "- Integrates data and policy information naturally\n"
        "- Highlights specific numbers and statistics where available\n"
        "- References specific protocols or guidelines by name\n"
        "- Is actionable and clear\n\n"
        "Response:"
    )
    chain = synth_prompt | llm | StrOutputParser()
    final = chain.invoke({"query": state["query"], "context": combined_context})

    return {"final_response": final}


def build_supervisor():
    global supervisor

    # Build and compile the Supervisor graph
    builder = StateGraph[SupervisorState, None, SupervisorState, SupervisorState](SupervisorState)

    # Add nodes
    builder.add_node("route_query", route_query)
    builder.add_node("execute_agents", execute_agents)
    builder.add_node("synthesize", synthesize_response)

    # Wire edges: linear flow
    builder.set_entry_point("route_query")
    builder.add_edge("route_query", "execute_agents")
    builder.add_edge("execute_agents", "synthesize")
    builder.add_edge("synthesize", END)

    # Compile
    supervisor = builder.compile()

    print("Supervisor graph compiled")
    print("Nodes: route_query -> execute_agents -> synthesize -> END")

    return supervisor


def run_supervisor(supervisor):
    # Test queries covering all three routes
    test_queries = [
        # Genie route (structured data)
        "How many disasters hit California in 2024?",
        "What was the total federal aid for hurricane-related disasters in 2024?",
        # Knowledge Assistant route (documents/policy)
        "What are FEMA's evacuation protocols for wildfire zones?",
        "Who is eligible for federal disaster assistance and what types of aid are available?",
        # Both routes (data + policy)
        "How many flood disasters occurred in 2024 and what are FEMA's flood response procedures?",
        "Which states had severity-5 disasters in 2024, and what safety protocols apply to those disaster types?",
    ]

    print(f"Running {len(test_queries)} test queries...\n")
    print("=" * 80)

    # Execute the supervisor and collect trace IDs
    trace_ids = []
    results_log = []

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 70)

        result = supervisor.invoke({"query": query})

        trace_id = mlflow.get_last_active_trace_id() or "N/A"
        trace_ids.append(trace_id)

        print(f"Route:     {result.get('route', 'N/A')}")
        print(f"Reasoning: {result.get('routing_reasoning', 'N/A')}")
        print(f"Response:  {result.get('final_response', 'N/A')[:300]}...")
        print(f"Trace ID:  {trace_id}")
        print("=" * 80)

        results_log.append({
            "query": query,
            "route": result.get("route"),
            "final_response": result.get("final_response"),
            "trace_id": trace_id,
        })

    print(f"\nCompleted {len(trace_ids)} queries. View traces in MLflow UI.")

    return trace_ids, results_log


def routing_summary(results_log):
    # Summary of routing decisions
    print("Routing Summary")
    print("=" * 60)
    for r in results_log:
        route_icon = {"genie": "[DATA]", "knowledge_assistant": "[DOCS]", "both": "[BOTH]"}
        icon = route_icon.get(r["route"], "[????]")
        print(f"  {icon} {r['query'][:60]}...")

    route_counts = Counter(r["route"] for r in results_log)
    print(f"\nRoute distribution: {dict(route_counts)}")


# Evaluation dataset with expected routes
eval_data = [
    {
        "inputs": {"query": "How many disasters hit California in 2024?"},
        "expectations": {"expected_route": "genie"},
    },
    {
        "inputs": {"query": "What are FEMA's wildfire safety guidelines?"},
        "expectations": {"expected_route": "knowledge_assistant"},
    },
    {
        "inputs": {"query": "What was the total federal aid for hurricanes in 2024?"},
        "expectations": {"expected_route": "genie"},
    },
    {
        "inputs": {"query": "What is the disaster declaration process?"},
        "expectations": {"expected_route": "knowledge_assistant"},
    },
    {
        "inputs": {"query": "How many tornado events occurred and what tornado safety procedures does FEMA recommend?"},
        "expectations": {"expected_route": "both"},
    },
    {
        "inputs": {"query": "Which state had the highest severity earthquake and what is the earthquake response protocol?"},
        "expectations": {"expected_route": "both"},
    },
]


def predict_fn(query: str) -> str:
    """Prediction function for mlflow.genai.evaluate()."""
    result = supervisor.invoke({"query": query})
    return result.get("final_response", "")


def layer1_builtin_scorers():
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

    # Define guidelines specific to a disaster response system
    disaster_guidelines = Guidelines(
        name="disaster_response_quality",
        guidelines=[
            "Responses about data should include specific numbers or statistics",
            "Responses about policies should reference specific protocol names or procedures",
            "Combined responses should clearly distinguish data findings from policy guidance",
            "All responses should be actionable and avoid vague generalities",
        ]
    )

    print(f"Evaluation dataset: {len(eval_data)} queries")
    print("Scorers: RelevanceToQuery, Safety, Guidelines (disaster_response_quality)")

    # Run built-in scorer evaluation
    with mlflow.start_run(run_name="Layer1-BuiltIn-Scorers"):
        builtin_results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                RelevanceToQuery(),
                Safety(),
                disaster_guidelines,
            ],
        )

    print("Layer 1 evaluation complete. View results in MLflow UI.")
    return builtin_results


@scorer
def routing_accuracy(inputs: dict, outputs, expectations: dict, trace) -> Feedback:
    """
    Evaluates whether the supervisor routed the query to the correct subagent.
    Compares the expected route (from eval dataset) against the actual route
    captured in the trace.
    """
    expected_route = expectations.get("expected_route", "")

    # Extract actual route from trace spans
    actual_route = None
    if trace and hasattr(trace, 'data') and hasattr(trace.data, 'spans'):
        for span in trace.data.spans:
            if span.name == "route_query":
                # The route_query node outputs include the route field
                span_outputs = span.outputs
                if isinstance(span_outputs, dict):
                    actual_route = span_outputs.get("route")
                break

    if actual_route is None:
        return Feedback(
            value=0.0,
            rationale="Could not extract routing decision from trace"
        )

    correct = actual_route == expected_route
    return Feedback(
        value=1.0 if correct else 0.0,
        rationale=f"Expected: {expected_route}, Actual: {actual_route}"
    )


def layer2_routing_accuracy():
    # Run routing accuracy evaluation
    with mlflow.start_run(run_name="Layer2-Routing-Accuracy"):
        routing_results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[routing_accuracy],
        )

    print("Layer 2 evaluation complete.")
    return routing_results


def layer3_agent_as_judge(model_name, trace_ids, results_log):
    from mlflow.genai.judges import make_judge

    # Judge model -- adjust based on your available endpoints
    # For Databricks: "databricks:/databricks-claude-sonnet-4"
    # For OpenAI via LiteLLM: "openai:/gpt-4o"
    JUDGE_MODEL = f"openai:/{model_name}"

    supervisor_judge = make_judge(
        name="supervisor_orchestration_quality",
        instructions=(
            "Analyze the multi-agent supervisor execution in {{ trace }}.\n\n"
            "Evaluate these dimensions:\n"
            "1. **Routing Decision**: Did the supervisor correctly identify whether the query "
            "needs structured data (Genie), document retrieval (Knowledge Assistant), or both?\n"
            "2. **Subagent Execution**: Did the invoked subagent(s) produce relevant, accurate results?\n"
            "3. **Synthesis Quality**: Did the final response coherently combine subagent outputs "
            "into an actionable answer?\n"
            "4. **Efficiency**: Were unnecessary subagents avoided? Was there redundancy?\n\n"
            "Rate the overall orchestration as: 'excellent', 'good', 'needs_improvement', or 'poor'.\n"
            "Provide specific evidence from the trace for your rating."
        ),
        model=JUDGE_MODEL,
    )

    print(f"Supervisor judge created (model: {JUDGE_MODEL})")

    # Run Agent-as-a-Judge evaluation on collected traces
    print("Running Agent-as-a-Judge evaluation...\n")
    print("=" * 70)

    judge_results = []
    for i, trace_id in enumerate(trace_ids):
        if trace_id == "N/A":
            print(f"Query {i+1}: No trace available, skipping")
            continue

        try:
            trace = mlflow.get_trace(trace_id)
            feedback = supervisor_judge(trace=trace)

            print(f"Query {i+1}: {results_log[i]['query'][:50]}...")
            print(f"  Rating: {feedback.value}")
            print(f"  Rationale: {feedback.rationale[:200]}...")
            print()

            judge_results.append({
                "query": results_log[i]["query"],
                "route": results_log[i]["route"],
                "rating": feedback.value,
                "rationale": feedback.rationale,
            })
        except Exception as e:
            print(f"Query {i+1}: Judge evaluation failed -- {e}")

    print("=" * 70)
    print(f"Evaluated {len(judge_results)} traces with Agent-as-a-Judge")

    return judge_results


def main():
    model_name, use_databricks_provider = setup()

    load_disaster_data()
    test_genie_agent()

    load_policy_documents()
    embed_documents()
    test_knowledge_assistant()

    supervisor = build_supervisor()
    trace_ids, results_log = run_supervisor(supervisor)
    routing_summary(results_log)

    layer1_builtin_scorers()
    layer2_routing_accuracy()
    layer3_agent_as_judge(model_name, trace_ids, results_log)


if __name__ == "__main__":
    main()
