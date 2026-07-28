import ollama

CHAT_MODEL = "qwen3:1.7b"
EMBED_MODEL = "bge-m3"

def test_chat():
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": "Say hello and confirm you are working in one short sentence."}
        ],
    )
    print("=== Chat Test ===")
    print(response["message"]["content"])
    print()

def test_embedding():
    text = "Ollama embeddings test with nomic embed text."
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text,
    )
    vec = response["embeddings"][0]
    print("=== Embedding Test ===")
    print(f"Input text: {text}")
    print(f"Vector length: {len(vec)}")
    print(f"First 8 values: {vec[:8]}")
    print()

if __name__ == "__main__":
    test_chat()
    test_embedding()