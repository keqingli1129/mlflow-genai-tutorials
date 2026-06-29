#!/usr/bin/env python
# coding: utf-8

"""Semantic Search over a PDF using LangChain, Chroma, and Ollama embeddings.

Required packages:
    pip install -U langchain langgraph langchain-chroma langchain-ollama \
        langchain-community pypdf

Pipeline:
    1. Load a PDF (URL or local path) into documents.
    2. Split documents into overlapping chunks.
    3. Embed chunks with a local Ollama model.
    4. Store embeddings in a persistent Chroma vector store.
    5. Query the store (similarity, scored, metadata-filtered).
    6. Build a retriever for RAG.
"""

import os
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
PDF_URL = "https://arxiv.org/pdf/2501.04040.pdf"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100  # 10% of chunk size
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "pdf_collection"
PERSIST_DIRECTORY = "./chroma_db"


def load_documents(pdf_url):
    """Load a PDF (URL or local path) into a list of page documents."""
    # STEP 1: DOCUMENTS AND DOCUMENT LOADERS
    loader = PyPDFLoader(pdf_url)
    documents = loader.load()

    print(f"\n✓ Loaded {len(documents)} pages from PDF")

    sample_doc = documents[0]
    print("\nSample Document Structure:")
    print(f"- Content length: {len(sample_doc.page_content)} characters")
    print(f"- Metadata: {sample_doc.metadata}")
    print(f"- Content preview: {sample_doc.page_content[:200]}...")

    return documents


def split_documents(documents):
    """Split page documents into overlapping character chunks."""
    # STEP 2: TEXT SPLITTING
    print("\n2.1 Configuring Text Splitter...")
    print(f"- Chunk size: {CHUNK_SIZE} characters")
    print(f"- Overlap: {CHUNK_OVERLAP} characters")
    print("- Method: Recursive character splitting")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,  # Preserves character index as metadata
    )

    print("\n2.2 Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"\n✓ Split {len(documents)} pages into {len(chunks)} chunks")

    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    print("\nChunk Analysis:")
    print(f"- Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"- Largest chunk: {max(chunk_sizes)} characters")
    print(f"- Smallest chunk: {min(chunk_sizes)} characters")

    return chunks


def create_embeddings():
    """Create the Ollama embedding function."""
    # STEP 3: EMBEDDINGS
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    print(f"\n✓ Embedding dimension: {len(embeddings.embed_query('Hello world'))}")
    return embeddings


def build_vector_store(chunks, embeddings):
    """Create a persistent Chroma vector store and add the chunks."""
    # STEP 4: VECTOR STORES
    print("\n4.1 Creating Chroma Vector Store...")
    print(f"- Collection name: {COLLECTION_NAME}")
    print(f"- Storage: {PERSIST_DIRECTORY}")
    print(f"- Embedding function: {EMBEDDING_MODEL} via Ollama")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    # Warm up the Ollama runner before the first real batch so its model
    # subprocess is fully started (avoids a connection-refused on cold start).
    embeddings.embed_query("warmup")

    # Add in small batches with a short retry. Sending all chunks in one
    # embed request can overload/crash the Ollama runner subprocess.
    batch_size = 50
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        for attempt in range(3):
            try:
                vector_store.add_documents(documents=batch)
                break
            except Exception as exc:  # noqa: BLE001 - retry transient runner errors
                if attempt == 2:
                    raise
                wait = 2 * (attempt + 1)
                print(f"  ! Batch {start}-{start + len(batch)} failed "
                      f"({exc}); retrying in {wait}s...")
                time.sleep(wait)
        print(f"  ✓ Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"✓ Added {len(chunks)} document chunks to vector store")

    return vector_store


def run_similarity_search(vector_store, query, k=5):
    """Basic similarity search using cosine similarity."""
    # STEP 5.1: BASIC SIMILARITY SEARCH
    print("\n5.1 Basic Similarity Search")
    print("Finding documents most similar to a query using cosine similarity...")

    results = vector_store.similarity_search(query, k=k)
    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(results)} most similar chunks:")

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Source: Page {doc.metadata.get('page', 'unknown')}")

    return results


def run_similarity_search_with_scores(vector_store, query, k=5):
    """Similarity search returning confidence scores."""
    # STEP 5.2: SIMILARITY SEARCH WITH SCORES
    print("\n5.2 Similarity Search with Scores")
    print("Same search but with similarity scores to see confidence levels...")

    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n--- Result {i} (Similarity Score: {score:.4f}) ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: Page {doc.metadata.get('page', 'unknown')}")

    return results_with_scores


def run_metadata_filtering(vector_store, chunks):
    """Demonstrate searching specific parts of the document via metadata."""
    # STEP 5.3: METADATA FILTERING
    print("\n5.3 Metadata Filtering")
    print("Using metadata filters to search specific parts of the document...")

    page_numbers = set()
    if chunks:
        sample_metadata = chunks[0].metadata
        print(f"\nSample metadata: {sample_metadata}")

        for chunk in chunks[:10]:  # Check first 10 chunks
            if "page" in chunk.metadata:
                page_numbers.add(chunk.metadata["page"])
        print(f"Available page numbers (sample): {sorted(page_numbers)[:5]}...")

    # 5.3.1 Filter by a specific page
    print("\n5.3.1 Filter by Specific Page")
    if page_numbers:
        target_page = sorted(page_numbers)[0]
        page_results = vector_store.similarity_search(
            "methodology approach",
            k=10,
            filter={"page": target_page},
        )
        print(f"Searching only in Page {target_page}:")
        for i, doc in enumerate(page_results, 1):
            print(
                f"  Result {i}: Page {doc.metadata.get('page')} - "
                f"{doc.page_content[:150]}..."
            )

    # 5.3.2 Multiple metadata filters
    print("\n5.3.2 Multiple Metadata Filters")
    complex_results = vector_store.similarity_search(
        "research findings",
        k=2,
        filter={
            "$and": [
                {"page": {"$lte": 10}},  # Page 10 or lower
                {"source": {"$ne": ""}},  # Has a source
            ]
        },
    )
    print("Using complex filter (page <= 10 AND has source):")
    for i, doc in enumerate(complex_results, 1):
        print(
            f"  Result {i}: Page {doc.metadata.get('page')} - "
            f"{doc.page_content[:150]}..."
        )


def build_retriever(vector_store, k=4):
    """Create a similarity retriever from the vector store."""
    # STEP 6: RETRIEVERS
    print("\n6. Creating Retriever...")
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_context(retriever, query):
    """Retrieve relevant chunks for a query (RAG foundation)."""
    # STEP 7: RAG FOUNDATION
    context_docs = retriever.invoke(query)
    print(f"\nQuery: '{query}'")
    print(f"✓ Retrieved {len(context_docs)} relevant document chunks")

    print("\nContext that would be sent to LLM:")
    for i, doc in enumerate(context_docs[:2], 1):  # Show first 2 for brevity
        print(f"\nChunk {i}: {doc.page_content[:250]}...")

    return context_docs


def main():
    """Run the full semantic search pipeline end to end."""
    documents = load_documents(PDF_URL)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = build_vector_store(chunks, embeddings)

    query = "What is the main methods available for RAG?"
    run_similarity_search(vector_store, query, k=5)
    run_similarity_search_with_scores(vector_store, query, k=5)
    run_metadata_filtering(vector_store, chunks)

    retriever = build_retriever(vector_store, k=4)
    final_query = "What are the main LLM models used for RAG?"
    retrieve_context(retriever, final_query)


if __name__ == "__main__":
    main()
