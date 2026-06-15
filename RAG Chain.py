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

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Configuration
LLM_MODEL = "qwen3:0.6b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "pdf_collection"
PERSIST_DIRECTORY = "./chroma_db"

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
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def ask(chain, question):
    """Ask the RAG chain a question and print the answer."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    answer = chain.invoke(question)
    print(f"\n💬 Answer:\n{answer}")
    return answer


def main():
    """Set up the RAG chain and run a sample query."""
    load_dotenv()

    llm = create_llm()
    embeddings = create_embeddings()
    vector_store = connect_vector_store(embeddings)

    chain = build_chain(llm, vector_store, k=4)
    ask(chain, "What are the main methods mentioned in this paper?")


if __name__ == "__main__":
    main()
