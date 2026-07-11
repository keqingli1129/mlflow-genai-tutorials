import os

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
    is_databricks_client,
)


def setup():
    load_dotenv()

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Determine provider and initialize client
    use_databricks_provider = is_databricks_client()
    use_databricks_ai_gateway = is_databricks_ai_gateway_client()

    if use_databricks_ai_gateway:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
        provider_name = "Databricks AI Gateway"
    else:
        client = get_openai_client()
        model_name = "gpt-5-mini"
        provider_name = "OpenAI"

    if not use_databricks_ai_gateway and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    print(f"✅ Environment configured: using {provider_name} client")
    print(f"   MLflow version: {mlflow.__version__}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Model name: {model_name}")

    # Enable OpenAI autologging
    mlflow.openai.autolog()

    mlflow.set_experiment("06-framework-integrations")

    print("✅ OpenAI autologging: ENABLED")

    return client, model_name, use_databricks_provider, use_databricks_ai_gateway


def langchain_simple_chain(model_name, use_databricks_provider, use_databricks_ai_gateway):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from utils.clnt_utils import (
        get_databricks_ai_gateway_langchain_client,
        get_databricks_langchain_chat_client,
        get_langchain_chat_openai_client,
    )

    # Enable LangChain autologging
    mlflow.langchain.autolog()

    print("✅ LangChain autologging enabled")

    # Simple LangChain chain
    print("\n🔗 LangChain Example 1: Simple Chain\n")

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are a {role}. Answer: {question}"
    )

    # Create LLM LangChain object based on provider
    if use_databricks_ai_gateway:
        llm = get_databricks_ai_gateway_langchain_client(model_name, temperature=1.0)
    elif use_databricks_provider:
        llm = get_databricks_langchain_chat_client(model_name, temperature=1.0)
    else:
        llm = get_langchain_chat_openai_client(model_name, temperature=1.0)

    # Create chain using LCEL (LangChain Expression Language)
    chain = prompt | llm | StrOutputParser()

    # Run chain (automatically traced!)
    # variables role and question are passed to the prompt template
    # during this invocation, the LLM will be called with the prompt template
    # and the variables will be substituted in the prompt template
    result = chain.invoke({
        "role": "MLflow expert",
        "question": "What makes LangChain different from using OpenAI directly?"
    })

    print(result)
    print("\n✅ Chain execution fully traced!")
    print("   - Prompt construction")
    print("   - LLM call")
    print("   - Output parsing")

    return llm


def langchain_multi_step_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # More complex chain with multiple steps
    print("\n🔗 LangChain Example 2: Multi-Step Chain\n")

    # Step 1: Generate topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a technical topic about {domain}"
    )
    topic_chain = topic_prompt | llm | StrOutputParser()

    # Step 2: Create outline
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a 3-point outline for: {topic}"
    )
    outline_chain = outline_prompt | llm | StrOutputParser()

    # Execute pipeline in sequence in Step 2.
    topic = topic_chain.invoke({"domain": "LLMOps"})
    print(f"Topic: {topic}\n")

    outline = outline_chain.invoke({"topic": topic})
    print(f"Outline:\n{outline}")

    print("\n✅ Multi-step chain traced!")
    print("   Each chain creates separate spans")
    print("   Full execution visible in MLflow UI")


def llamaindex_document_qa(use_databricks_ai_gateway):
    import os

    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

    # Enable LlamaIndex autologging
    mlflow.llama_index.autolog()

    # Configure LlamaIndex Settings based on provider
    # LlamaIndex defaults to OpenAI, so we need to explicitly configure it for Databricks AI Gateway
    if use_databricks_ai_gateway:
        # Get Databricks AI Gateway credentials
        databricks_token = os.environ.get("DATABRICKS_TOKEN")
        ai_gateway_base_url = os.environ.get("AI_GATEWAY_BASE_URL")

        # Configure LLM for Databricks AI Gateway (OpenAI-compatible endpoint)
        Settings.llm = LlamaIndexOpenAI(
            model=os.environ.get("AI_GATEWAY_LLM_MODEL", "jsd-gpt-5-2"),
            api_key=databricks_token,
            api_base=ai_gateway_base_url
        )

        # Configure embedding model for Databricks AI Gateway
        # Use model_name (not model) to bypass OpenAIEmbeddingModelType enum validation
        Settings.embed_model = OpenAIEmbedding(
            model_name=os.environ.get("AI_GATEWAY_EMBED_MODEL", "jsd-text-embedding-3-small"),
            api_key=databricks_token,
            api_base=ai_gateway_base_url
        )
        print("✅ LlamaIndex configured for Databricks AI Gateway")
        print(f"   LLM: {Settings.llm.model}")
        print(f"   Embeddings: {Settings.embed_model.model_name}")
    else:
        # Use default OpenAI settings (requires OPENAI_API_KEY)
        print("✅ LlamaIndex using default OpenAI configuration")

    print("✅ LlamaIndex autologging enabled")

    # Create sample documents. Normally, you get PDFs, markdown files, HTML, or text files.
    # For this example, we'll just use text for simplicity.

    print("\n📚 LlamaIndex Example: Document Q&A\n")

    documents = [
        Document(text="MLflow is an open source AI platform for the complete GenAI lifecycle. It provides experiment tracking, prompt registry, and agent evaluationc apabilities."),
        Document(text="MLflow Tracing captures the complete execution of GenAI applications, including LLM calls, retrieval steps, and tool usage."),
        Document(text="MLflow integrates with 30+ frameworks including OpenAI, LangChain, LlamaIndex, and more."),
        Document(text="MLflow supports collaborative development with experiment sharing, prompt management and versioning."),
        Document(text="MLflow is open source and supported by Databricks. It's also OpenTelemetry-compatible, so you can monitor in production without vendor lock-in."),
    ]

    # Create index (automatically traced)
    # Note: This uses the LLM and embedding model configured in Settings above
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine associated with the index
    query_engine = index.as_query_engine()

    # Query the index (automatically traced)
    response = query_engine.query("What tracing capabilities does MLflow have?")

    print("Query: What tracing capabilities does MLflow have?")
    print(f"\nAnswer: {response}")

    print("\n✅ LlamaIndex execution fully traced!")
    print("   - Document indexing")
    print("   - Query embedding")
    print("   - Retrieval")
    print("   - Response synthesis")


def langgraph_customer_triage(llm):
    from typing import Literal, TypedDict

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, StateGraph

    # ── Shared state ─────────────────────────────────────────────────────────────
    class CustomerServiceState(TypedDict):
        message: str
        category: str    # filled by classify node
        response: str    # filled by respective handler node

    # ── Node 1: classify agent for the incoming message ────────────────────────────────────
    def classify(state: CustomerServiceState) -> CustomerServiceState:
        prompt = ChatPromptTemplate.from_template(
            "You are a customer service triage agent. "
            "Classify the following message into exactly one of this category: "
            "'billing', 'tech_support', or 'customer_inquiry'. "
            "Reply with one word only.\n\nMessage: {message}"
        )
        # llm is a LangChain object
        chain = prompt | llm | StrOutputParser()
        category = chain.invoke({"message": state["message"]}).strip().lower()
        # Normalize to a known category
        if "billing" in category:
            category = "billing"
        elif "tech" in category:
            category = "tech_support"
        else:
            category = "customer_inquiry"
        return {"category": category}

    # ── Node 2a: billing agent specialist ──────────────────────────────────────────────
    def handle_billing(state: CustomerServiceState) -> CustomerServiceState:
        prompt = ChatPromptTemplate.from_template(
            "You are a billing specialist. Help the customer with their billing issue.\n\n"
            "Customer message: {message}"
        )
        chain = prompt | llm | StrOutputParser()
        return {"response": chain.invoke({"message": state["message"]})}

    # ── Node 2b: technical support engineer agent ──────────────────────────────────────
    def handle_tech_support(state: CustomerServiceState) -> CustomerServiceState:
        prompt = ChatPromptTemplate.from_template(
            "You are a technical support engineer. Help the customer resolve their technical issue.\n\n"
            "Customer message: {message}"
        )
        chain = prompt | llm | StrOutputParser()
        return {"response": chain.invoke({"message": state["message"]})}

    # ── Node 2c: customer success agent ──────────────────────────────────────────
    def handle_customer_inquiry(state: CustomerServiceState) -> CustomerServiceState:
        prompt = ChatPromptTemplate.from_template(
            "You are a customer success agent. Answer the customer's question helpfully.\n\n"
            "Customer message: {message}"
        )
        chain = prompt | llm | StrOutputParser()
        return {"response": chain.invoke({"message": state["message"]})}

    # ── Routing function ──────────────────────────────────────────────────────────
    def route(state: CustomerServiceState) -> Literal["handle_billing", "handle_tech_support", "handle_customer_inquiry"]:
        routes = {
            "billing": "handle_billing",
            "tech_support": "handle_tech_support",
            "customer_inquiry": "handle_customer_inquiry",
        }
        return routes.get(state["category"], "handle_customer_inquiry")

    # ── Build and compile the graph ───────────────────────────────────────────────
    # Create a StateGraph builder -- specialized for type-checking so that the state is CustomerServiceState,
    # there's no context, and both input and output are CustomerServiceState -- and pass CustomerServiceState
    # as the actual state schema

    builder = StateGraph[CustomerServiceState, None, CustomerServiceState, CustomerServiceState](CustomerServiceState)
    builder.add_node("classify", classify)
    builder.add_node("handle_billing", handle_billing)
    builder.add_node("handle_tech_support", handle_tech_support)
    builder.add_node("handle_customer_inquiry", handle_customer_inquiry)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route)
    builder.add_edge("handle_billing", END)
    builder.add_edge("handle_tech_support", END)
    builder.add_edge("handle_customer_inquiry", END)

    app = builder.compile()

    # ── Run sample messages (one per route) ───────────────────────────────────────
    print("🔀 LangGraph Example: Customer Service Triage\n")

    test_messages = [
        "I was charged twice for my subscription this month.",
        "The app crashes whenever I try to export a report.",
        "Can you explain the difference between the Pro and Enterprise plans?",
    ]

    for msg in test_messages:
        result = app.invoke({"message": msg})
        print(f"Message  : {result['message']}")
        print(f"Category : {result['category']}")
        print(f"Response : {result['response'][:200]}...")
        print()

    print("✅ LangGraph execution fully traced!")
    print("   🔍 View in MLflow UI — each invocation produces a hierarchical trace:")
    print("      - Root span: full graph invocation")
    print("      - 'classify' span: LLM call that picks the route")
    print("      - 'handle_billing' / 'handle_tech_support' / 'handle_customer_inquiry': routed handler")


def main():
    client, model_name, use_databricks_provider, use_databricks_ai_gateway = setup()

    llm = langchain_simple_chain(model_name, use_databricks_provider, use_databricks_ai_gateway)
    langchain_multi_step_chain(llm)

    llamaindex_document_qa(use_databricks_ai_gateway)

    langgraph_customer_triage(llm)


if __name__ == "__main__":
    main()
