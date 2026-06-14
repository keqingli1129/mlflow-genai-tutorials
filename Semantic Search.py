#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required packages:
# pip install -U langchain langgraph langchain-chroma langchain-ollama langchain-community pypdf

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[3]:


# ============================================================================
# STEP 1: DOCUMENTS AND DOCUMENT LOADERS
# ============================================================================
# Load PDF - works with both online URLs and local file paths
pdf_url = "https://arxiv.org/pdf/2501.04040.pdf"
loader = PyPDFLoader(pdf_url)
documents = loader.load()


# In[4]:


documents[0]
print(f"\n✓ Loaded {len(documents)} pages from PDF")


# In[5]:


sample_doc = documents[0]
print(f"\nSample Document Structure:")
print(f"- Content length: {len(sample_doc.page_content)} characters")
print(f"- Metadata: {sample_doc.metadata}")
print(f"- Content preview: {sample_doc.page_content[:200]}...")


# In[6]:


# ============================================================================
# STEP 2: TEXT SPLITTING  
# ============================================================================
print("\n2.1 Configuring Text Splitter...")
print("- Chunk size: 1024 characters (as specified)")
print("- Overlap: 100 characters (10% overlap)")
print("- Method: Recursive character splitting")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100,  # 10% of 1024
    length_function=len,
    add_start_index=True,  # Preserves character index as metadata
)

print("\n2.2 Splitting documents into chunks...")
chunks = text_splitter.split_documents(documents)

print(f"\n✓ Split {len(documents)} pages into {len(chunks)} chunks")

chunk_sizes = [len(chunk.page_content) for chunk in chunks]
print(f"\nChunk Analysis:")
print(f"- Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
print(f"- Largest chunk: {max(chunk_sizes)} characters")
print(f"- Smallest chunk: {min(chunk_sizes)} characters")


# In[11]:


# ============================================================================
# STEP 3: EMBEDDINGS
# ======
# 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


# In[10]:


len(embeddings.embed_query("Hello world"))


# In[12]:


# ============================================================================
# STEP 4: VECTOR STORES
# ============================================================================
print("\n4.1 Creating Chroma Vector Store...")
print("- Collection name: pdf_collection")
print("- Storage: Local persistent directory")
print("- Embedding function: nomic-embed-text via Ollama")

vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

vector_store.add_documents(documents=chunks)

print(f"✓ Added {len(chunks)} document chunks to vector store")


# In[13]:


# ============================================================================
# STEP 5: QUERYING THE VECTOR STORE
# ============================================================================

print("\n5.1 Basic Similarity Search")
print("Finding documents most similar to a query using cosine similarity...")

query = "What is the main methods available for RAG?"
results = vector_store.similarity_search(query, k=5)

print(f"\nQuery: '{query}'")
print(f"Retrieved {len(results)} most similar chunks:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Content: {doc.page_content[:300]}...")
    print(f"Source: Page {doc.metadata.get('page', 'unknown')}")


# In[14]:


print("\n5.2 Similarity Search with Scores")
print("Same search but with similarity scores to see confidence levels...")

results_with_scores = vector_store.similarity_search_with_score(query, k=5)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n--- Result {i} (Similarity Score: {score:.4f}) ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: Page {doc.metadata.get('page', 'unknown')}")


# In[17]:


print("\n5.3 Metadata Filtering")
print("Using metadata filters to search specific parts of the document...")

# First, let's see what metadata is available
print("\nAvailable metadata in our chunks:")
if chunks:
    sample_metadata = chunks[0].metadata
    print(f"Sample metadata: {sample_metadata}")

    # Get unique page numbers for filtering examples
    page_numbers = set()
    for chunk in chunks[:10]:  # Check first 10 chunks
        if 'page' in chunk.metadata:
            page_numbers.add(chunk.metadata['page'])
    print(f"Available page numbers (sample): {sorted(list(page_numbers))[:5]}...")


# In[18]:


print("\n5.3.1 Filter by Specific Page")
if page_numbers:
    target_page = sorted(list(page_numbers))[0]  # Use first available page
    page_results = vector_store.similarity_search(
        "methodology approach",
        k=10,
        filter={"page": target_page}
    )
    print(f"Searching only in Page {target_page}:")
    for i, doc in enumerate(page_results, 1):
        print(f"  Result {i}: Page {doc.metadata.get('page')} - {doc.page_content[:150]}...")


# In[19]:


print("\n5.3.3 Multiple Metadata Filters")
# Complex filtering with multiple conditions
complex_results = vector_store.similarity_search(
    "research findings",
    k=2,
    filter={
        "$and": [
            {"page": {"$lte": 10}},  # Page 0 or higher
            {"source": {"$ne": ""}}  # Has a source
        ]
    }
)

print("Using complex filter (page >= 0 AND has source):")
for i, doc in enumerate(complex_results, 1):
    print(f"  Result {i}: Page {doc.metadata.get('page')} - {doc.page_content[:150]}...")


# In[20]:


# ============================================================================
# STEP 6: RETRIEVERS
# ============================================================================
print("\n6. Creating Retriever...")

# Similarity Retriever
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# In[21]:


# ============================================================================
# STEP 7: RAG FOUNDATION
# ============================================================================
# final_query = "What are the key contributions of this paper?"
final_query = "What are the main LLM models used for RAG?"
context_docs = similarity_retriever.invoke(final_query)

print(f"\nQuery: '{final_query}'")
print(f"✓ Retrieved {len(context_docs)} relevant document chunks")


# In[22]:


# Show what would be sent to LLM
print(f"\nContext that would be sent to LLM:")
for i, doc in enumerate(context_docs[:2], 1):  # Show first 2 for brevity
    print(f"\nChunk {i}: {doc.page_content[:250]}...")

