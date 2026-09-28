import chromadb
import ollama


EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="company_policy"
)


print("RAG chat ready. Type 'exit' or 'quit' to stop.")

while True:
    question = input("\nAsk a question: ").strip()

    if not question:
        continue

    if question.lower() in ("exit", "quit"):
        break

    question_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    # print("\n--- Retrieved Context ---")
    # print(context)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say "I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("\n--- Answer ---")
    print(answer)
