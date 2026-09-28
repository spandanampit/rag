import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")
collection=client.get_collection(name="company_policy")
question = "How many vacation days can I carry forward?"


# Create embedding for question
question_embedding = ollama.embed(
    model=EMBED_MODEL,
    input=question
)["embeddings"][0]

# search
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3 #returns 3 closest chunks
)

# dispay
print("\nQuestion:")
print(question)

print("\nRelevant chunks:")

for i, document in enumerate(results["documents"][0]):

    distance = results["distances"][0][i]

    print(f"\n--- Result {i + 1} ---")
    print("Distance:", distance)
    print("Document:", document)