#!/usr/bin/env python
# coding: utf-8

"""Agentic RAG over a Chroma vector store using LangChain agents and Ollama.

Required packages:
    pip install -U langchain langgraph langchain-chroma langchain-ollama python-dotenv

Assumes the Chroma collection has already been populated (see "Semantic Search.py").

Pipeline:
    1. Set up the Ollama LLM and embeddings.
    2. Connect to the existing Chroma vector store.
    3. Define a retrieval tool over the vector store.
    4. Build a LangChain agent that uses the tool.
    5. Query the agent (single question, or interactive chat).
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Configuration
LLM_MODEL = "qwen3:0.6b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "pdf_collection"
PERSIST_DIRECTORY = "./chroma_db"

AGENT_PROMPT = """You are a research assistant with a document retrieval tool.

Tool:
- retrieve_context: Search the document for relevant information

Always use the tool to find relevant information before answering.
Cite page numbers and be thorough."""


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


def verify_vector_store(vector_store, test_query="methodology"):
    """Sanity-check the vector store connection with a sample search."""
    print("\n🔍 Testing Vector Store Connection...")

    collection = vector_store._collection
    print(f"✓ Collection '{collection.name}' found with {collection.count()} documents")

    results = vector_store.similarity_search(test_query, k=3)
    print(f"\n✓ Sample search for '{test_query}':")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")


def make_retrieve_tool(vector_store):
    """Build the retrieval tool bound to the given vector store."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant information from the document to answer the query."""
        print(f"🔍 Searching: '{query}'")

        docs = vector_store.similarity_search(query, k=4)
        content = "\n\n".join(
            f"Page {doc.metadata.get('page', '?')}: {doc.page_content}"
            for doc in docs
        )

        print(f"✓ Found {len(docs)} relevant chunks")
        return content, docs

    return retrieve_context


def build_agent(llm, vector_store):
    """Create the agentic RAG with a single retrieval tool."""
    tools = [make_retrieve_tool(vector_store)]
    return create_agent(llm, tools, system_prompt=AGENT_PROMPT)


def ask(rag_agent, question):
    """Ask the agentic RAG a question and stream the response."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    for event in rag_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        msg = event["messages"][-1]

        # Show tool usage
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n🔧 Using: {tc['name']} with {tc['args']}")
        # Show final answer
        elif hasattr(msg, "content") and msg.content:
            print(f"\n💬 Answer:\n{msg.content}")


def chat(rag_agent):
    """Start an interactive chat with the agentic RAG."""
    print("\n🤖 Agentic RAG Chat - Type 'quit' to exit")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if question:
            ask(rag_agent, question)


def main():
    """Set up the agentic RAG and run a sample query plus interactive chat."""
    load_dotenv()

    llm = create_llm()
    embeddings = create_embeddings()
    vector_store = connect_vector_store(embeddings)

    verify_vector_store(vector_store)

    rag_agent = build_agent(llm, vector_store)

    # Sample query
    ask(rag_agent, "What are the main methods mentioned in this paper?")

    # Interactive chat
    chat(rag_agent)


if __name__ == "__main__":
    main()
