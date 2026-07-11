import time

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)

# Shared across the traced functions below (set in setup())
client = None
model_name = None


def setup():
    global client, model_name

    # Load environment
    load_dotenv()

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
    else:
        # Initialize OpenAI
        client = get_openai_client()
        model_name = "gpt-5-mini"
    print("✅ Environment configured: using", "Databricks" if use_databricks_provider else "OpenAI", "client")
    print(f"   MLflow version: {mlflow.__version__}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")

    # Create experiment
    mlflow.set_experiment("07-manual-tracing")

    print("✅ Environment configured")
    print("   Ready for manual tracing!")

    return client, model_name, use_databricks_provider


def span_types_demo():
    from typing import List

    # Every function uses @mlflow.trace(name="...", span_type="...")

    @mlflow.trace(name="db_query_simulation", span_type="CHAIN")
    def simulate_custom_query_processing(query: str) -> str:
        """Simulate query preprocessing with an in-memory SQLite DB — CHAIN span."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        # create a table
        conn.execute("CREATE TABLE q (id INTEGER PRIMARY KEY, raw TEXT, processed TEXT)")
        # process the query
        processed = query.strip().lower()
        # insert the query into the table
        conn.execute("INSERT INTO q (raw, processed) VALUES (?, ?)", (query, processed))
        # commit the transaction
        conn.commit()
        # simulate latency
        time.sleep(0.3)  # Simulate latency
        row = conn.execute("SELECT processed FROM q WHERE id = 1").fetchone()
        conn.close()
        return row[0]

    @mlflow.trace(name="preprocess_query", span_type="PARSER")
    def preprocess_query(user_query: str) -> str:
        """Parse/clean a query — PARSER span wrapping a CHAIN child span."""
        return simulate_custom_query_processing(user_query)

    @mlflow.trace(name="document_retriever", span_type="RETRIEVER")
    def retrieve_documents(query: str, top_k: int = 3) -> List[str]:
        """Simulate vector DB retrieval — RETRIEVER span."""

        # simulate latency You would use a real vector DB here
        time.sleep(0.3)
        # simulate retrieval
        docs = [
            "MLflow is an open source platform for the ML and GenAI Applications.",
            "MLflow Tracing provides end-to-end observability for GenAI applications.",
            "MLflow supports experiment tracking, model registry, prompt management, and AI Gateway."
        ]
        return docs[:top_k]

    @mlflow.trace(name="get_embedding", span_type="LLM")
    def get_embedding(query: str) -> List[float]:
        """Call OpenAI embeddings API — LLM span."""
        response = client.embeddings.create(
            input=query.replace("\n", " "), model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @mlflow.trace(name="query_embedder", span_type="EMBEDDING")
    def embed_query(query: str) -> List[float]:
        """Embed a query — EMBEDDING parent span with LLM child span."""
        return get_embedding(query)

    print("\n🔍 Testing manual tracing and span types...\n")

    # Plain trace: shows nested spans (preprocess_query → simulate_custom_query_processing)
    result = preprocess_query("  What is MLflow?  ")
    print(f"preprocess_query output: {result!r}")

    query = "What are MLflow's tracing capabilities?"

    # Typed span: EMBEDDING parent → LLM child
    embedding = embed_query(query)
    print(f"\nembed_query output: {embedding[:3]}...")

    # Typed span: RETRIEVER
    documents = retrieve_documents(query, top_k=2)
    print(f"\nretrieve_documents output: {len(documents)} docs")
    for i, doc in enumerate(documents, 1):
        print(f"  Doc {i}: {doc[:80]}...")

    print("\n✅ All spans created!")
    print("""
🔍 In MLflow UI you'll see distinct span types per function:
  preprocess_query          → PARSER
    db_query_simulation     → CHAIN          nested child span
  query_embedder            → EMBEDDING
    get_embedding           → LLM            nested child span
  document_retriever        → RETRIEVER
""")


def custom_attributes_demo():
    from typing import List

    # Add custom attributes to spans

    @mlflow.trace(name="enhanced_retriever", span_type="RETRIEVER")
    def enhanced_retriever(query: str, top_k: int = 3) -> List[str]:
        """
        Retriever with rich metadata logging.
        """
        span = mlflow.get_current_active_span()
        # set attributes
        span.set_attributes({"top_k":top_k,
                            "query_length":len(query),
                            "search_method":"cosine_similarity"})

        # Simulate retrieval. Substitute this with a real vector database call.
        time.sleep(0.4)
        docs = [

            "MLflow provides tracing capabilities for end-to-end Agentic workflows"
            "MLflow logs all operations and events in a trace, providing a complete view of the workflow."
            "MLflow supports multiple agent frameworks like LangChain, LlamaIndex, and more for tracing"
        ]

        # set attributes

        span.set_attributes({"num_results": len(docs[:top_k]),
                            "total_docs_in_db": 1000})  # Simulated

        return docs[:top_k]

    # Test it
    print("\n📊 Testing custom attributes...\n")
    query = "What is MLflow tracing?"
    docs = enhanced_retriever(query, top_k=2)

    print("\n✅ Span created with custom attributes!")
    print("\n🔍 Custom attributes visible in UI:")
    print("   - top_k: 2")
    print(f"   - query_length: {len(query)}")
    print("   - search_method: vector_similarity")
    print("   - num_results: 2")
    print("   - total_docs_in_db: 1000")


def agentic_workflow_demo():
    # Agentic workflow with tool usage
    from typing import Dict
    from typing import List
    import json

    @mlflow.trace(name="weather_tool", span_type="TOOL")
    def get_weather(city: str) -> Dict:
        """Simulated weather tool — replace with a real weather API."""
        span = mlflow.get_current_active_span()
        span.set_attributes({"tool_name": "weather_api", "city": city, "api_version": "v2.0"})

        # simulate latency
        # You would use a real weather API here
        time.sleep(0.2)

        # simulate weather data
        weather_data = {"city": city, "temperature": "72°F", "condition": "Sunny", "humidity": "45%"}
        span.set_attributes({"data_retrieved": True})
        return weather_data

    @mlflow.trace(name="create_calendar_invite", span_type="TOOL")
    def create_calendar_invite(title: str, date: str, start_time: str, attendees: List[str]) -> Dict:
        """Simulated calendar invite tool — replace with Google Calendar / Outlook API."""
        span = mlflow.get_current_active_span()
        span.set_attributes({
            "tool_name": "create_calendar_invite",
            "title": title,
            "date": date,
            "start_time": start_time,
            "num_attendees": len(attendees)
        })
        time.sleep(0.1)
        return {"status": "created", "invite_id": "inv-20240301-001",
                "title": title, "date": date, "start_time": start_time, "attendees": attendees}

    @mlflow.trace(name="send_email", span_type="TOOL")
    def send_email(to: str, subject: str, body: str) -> Dict:
        """Simulated email tool — replace with SendGrid / SES / SMTP."""
        span = mlflow.get_current_active_span()
        span.set_attributes({"tool_name": "send_email", "to": to, "subject": subject, "body_length": len(body)})
        time.sleep(0.1)
        return {"status": "sent", "message_id": "msg-20240301-001", "to": to, "subject": subject}

    @mlflow.trace(name="generate_random_password", span_type="TOOL")
    def generate_random_password(length: int) -> str:
        """Generate a cryptographically strong random password."""
        import secrets
        import string

        span = mlflow.get_current_active_span()
        span.set_attributes({"tool_name": "generate_random_password", "length": length})
        if length < 8:
            raise ValueError("Password length must be at least 8 characters")
        lower, upper, digits, special = (
            string.ascii_lowercase, string.ascii_uppercase,
            string.digits, "!@#$%^&*()_+-=[]{}|;:,.<>?"
        )
        password = [secrets.choice(lower), secrets.choice(upper),
                    secrets.choice(digits), secrets.choice(special)]
        all_chars = lower + upper + digits + special
        password += [secrets.choice(all_chars) for _ in range(length - 4)]
        secrets.SystemRandom().shuffle(password)
        return "".join(password)

    @mlflow.trace(name="agent_planning", span_type="AGENT")
    def agent_plan(user_query: str) -> Dict:
        """LLM decides which tool to use and extracts parameters."""

        span = mlflow.get_current_active_span()
        span.set_attributes({"user_query": user_query})

        # Define the system prompt
        system_prompt = """You are an agent with access to:
1. get_weather(city) - Get weather information
2. create_calendar_invite(title, date, start_time, attendees) - Create a calendar invite; attendees is a list of email strings
3. send_email(to, subject, body) - Send an email
4. generate_random_password(length) - Generate a random password

Decide which tool to use and extract parameters.
Respond with JSON: {"tool": "tool_name", "params": {...}}"""
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_query}],
            response_format={"type": "json_object"},
        )
        plan = json.loads(response.choices[0].message.content)
        span.set_attributes({"selected_tool": plan["tool"]})
        return plan

    @mlflow.trace(name="agent_execution", span_type="CHAIN")
    def run_agent(user_query: str) -> str:
        """Parent CHAIN: plan → execute tool → generate final response."""
        span = mlflow.get_current_active_span()
        span.set_attributes({"agent_version": "v1.0", "user_query": user_query})

        # Step 1: LLM picks the right tool
        plan = agent_plan(user_query)

        # Step 2: Execute the selected tool
        if plan["tool"] == "get_weather":
            tool_result = get_weather(plan["params"]["city"])
        elif plan["tool"] == "create_calendar_invite":
            tool_result = create_calendar_invite(
                plan["params"]["title"], plan["params"]["date"],
                plan["params"]["start_time"], plan["params"]["attendees"])
        elif plan["tool"] == "send_email":
            tool_result = send_email(
                plan["params"]["to"], plan["params"]["subject"], plan["params"]["body"])
        elif plan["tool"] == "generate_random_password":
            tool_result = generate_random_password(plan["params"]["length"])
        else:
            raise ValueError(f"Unknown tool: {plan['tool']}")

        span.set_attributes({"tool_executed": plan["tool"]})

        # Step 3: LLM summarises the tool result
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user",
                       "content": f"Tool result: {tool_result}. Answer the user's original request: {user_query}"}],
            max_completion_tokens=1000,
        )
        final_answer = response.choices[0].message.content
        span.set_attributes({"answer_generated": True})
        return final_answer

    # Test the agent with four realistic queries
    print("\n🤖 Testing agentic workflow with tool tracing...\n")

    query_1 = "What's the weather in San Francisco?"
    answer_1 = run_agent(query_1)
    print(f"Query: {query_1}")
    print(f"Answer: {answer_1}\n")

    query_2 = "Schedule a team meeting called 'Q2 Planning' on 2026-04-01 at 2pm with alice@example.com and bob@example.com"
    answer_2 = run_agent(query_2)
    print(f"Query: {query_2}")
    print(f"Answer: {answer_2}\n")

    query_3 = "Send an email to alice@example.com with subject 'Meeting Notes' and body 'Please find the Q2 planning notes attached.'"
    answer_3 = run_agent(query_3)
    print(f"Query: {query_3}")
    print(f"Answer: {answer_3}\n")

    query_4 = "Generate a random password with 12 characters"
    answer_4 = run_agent(query_4)
    print(f"Query: {query_4}")
    print("Answer: ************")

    print("\n" + "="*60)
    print("✅ Agent Workflow Fully Traced!")
    print("="*60)

    print("""
\n🔍 Agent Trace Hierarchy:

agent_execution (CHAIN)
├── agent_planning (AGENT)
│   └── OpenAI API call (auto-traced)
├── get_weather | create_calendar_invite | send_email | generate_random_password (TOOL)
└── OpenAI API call for final response (auto-traced)

Key insights visible:
✓ Which tool was selected by the LLM
✓ Tool execution time and parameters
✓ Tool result logged as span output
✓ Total agent reasoning time
✓ Token usage across all LLM calls
""")


def rag_pipeline_demo():
    from typing import Dict, List
    import json

    @mlflow.trace(name="parse_query", span_type="PARSER")
    def parse_query(query: str) -> Dict:
        """Use an LLM to extract intent and entities — traced as a PARSER span."""
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content":
                f'Return JSON with "intent" (single word) and "entities" (list) for: "{query}"'}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {"intent": result.get("intent", "question"),
                "entities": result.get("entities", []),
                "cleaned_query": query.strip()}

    @mlflow.trace(name="embed_query", span_type="EMBEDDING")
    def embed_query(query: str) -> List[float]:
        """Call the OpenAI embeddings API — traced as an EMBEDDING span."""
        response = client.embeddings.create(
            input=query.replace("\n", " "), model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @mlflow.trace(name="vector_search", span_type="RETRIEVER")
    def vector_search(embedding: List[float], top_k: int = 3) -> List[str]:
        """Simulate a vector DB lookup — traced as a RETRIEVER span."""
        span = mlflow.get_current_active_span()
        span.set_attributes({"embedding_dim": len(embedding), "top_k": top_k})

        # Simulate retrieval. Substitute this with a real vector database call.
        time.sleep(0.15)
        docs = [
            "MLflow is an open source platform for managing ML and GenAI lifecycle.",
            "MLflow Tracing captures LLM execution with detailed spans.",
            "MLflow integrates with OpenAI, LangChain, and 30+ frameworks.",
        ]
        return docs[:top_k]

    @mlflow.trace(name="format_prompt", span_type="PARSER")
    def format_prompt(query: str, docs: List[str]) -> str:
        """Build the RAG prompt with retrieved context — traced as a PARSER span."""
        context = "\n".join(f"- {doc}" for doc in docs)
        return f"Use this context to answer:\n\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"

    @mlflow.trace(name="generate_answer", span_type="LLM")
    def generate_answer(prompt: str) -> str:
        """Call the LLM to produce a final answer — traced as an LLM span."""
        span = mlflow.get_current_active_span()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        answer = response.choices[0].message.content
        if response.usage:
            span.set_attributes({"tokens_used": response.usage.total_tokens})
        return answer

    @mlflow.trace(name="rag_qa_pipeline", span_type="CHAIN")
    def rag_qa_pipeline(user_query: str, top_k: int = 3) -> Dict:
        """Parent CHAIN span — wraps the full pipeline end-to-end."""
        span = mlflow.get_current_active_span()
        span.set_attributes({"pipeline_version": "v1.0", "user_query": user_query})

        # start the pipeline as a CHAIN span
        #  Step 1: Parse the query
        parsed    = parse_query(user_query)

        # Step 2: Embed the query
        embedding = embed_query(parsed["cleaned_query"])

        # Step 3: Search the vector database
        docs      = vector_search(embedding, top_k=top_k)

        # Step 4: Format the prompt
        prompt    = format_prompt(user_query, docs)

        # Step 5: Generate the answer
        answer    = generate_answer(prompt)

        span.set_attributes({"num_docs": len(docs), "answer_generated": True})
        return {"query": user_query, "answer": answer, "num_docs": len(docs)}

    # Test the complete RAG pipeline
    print("\n🤖 Testing complete RAG pipeline...\n")

    result = rag_qa_pipeline("What tracing capabilities does MLflow provide?", top_k=3)

    print(f"Query:  {result['query']}")
    print(f"\nDocs used: {result['num_docs']}")
    print(f"\nAnswer:\n{result['answer']}")

    print("\n" + "="*60)
    print("✅ Full RAG pipeline traced!")
    print("="*60)
    print("""
🔍 Trace hierarchy in MLflow UI:

rag_qa_pipeline  (CHAIN)
├── parse_query  (PARSER)
├── embed_query  (EMBEDDING)
├── vector_search  (RETRIEVER)
├── format_prompt  (PARSER)
└── generate_answer  (LLM)

Each span shows inputs, outputs, duration, and custom attributes.
""")


def debugging_demo():
    from typing import Dict

    # Debugging example: Trace a buggy function, which can raise an error

    @mlflow.trace(name="buggy_processor", span_type="PARSER")
    def process_with_validation(data: Dict) -> Dict:
        """
        Fake function with potential errors.
        """
        span = mlflow.get_current_active_span()
        span.set_attributes({
            "input_keys": list(data.keys())
        })

        # Log intermediate state for debugging
        span.set_attributes({
            "validation_step": "checking_required_fields"
        })

        # This might raise an error
        if "required_field" not in data:
            span.set_attributes({
                "error_type": "missing_field"
            })
            raise ValueError("Missing required_field")
        span.set_attributes({
            "validation_step": "passed"
        })

        # Process data
        result = {"processed": True, "value": data.get("required_field")}

        return result

    print("\n🐛 Testing error tracing...\n")

    # Success case
    try:
        result = process_with_validation({"required_field": "test"})
        print(f"✅ Success case: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Failure case
    print("\nTesting failure case...")
    try:
        result = process_with_validation({"wrong_field": "test"})
        print(f"✅ Success case: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 Error details captured in trace!")

    print("""
\n💡 Debugging with Traces:

1. Filter traces by error status in UI
2. View the failed span (marked in red)
3. Check custom attributes:
   - What were the inputs?
   - What validation step failed?
   - What was the error type?
4. Compare with successful traces
5. Reproduce the issue with exact inputs
6. Use MLflow Assistant to get insights on the failed traces and root cause

Attributes logged BEFORE the error help identify root cause!
""")


def main():
    setup()
    span_types_demo()
    custom_attributes_demo()
    agentic_workflow_demo()
    rag_pipeline_demo()
    debugging_demo()


if __name__ == "__main__":
    main()
