import ollama


text = "Employees can carry forward a maximum of 5 unused vacation days."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Embedding dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])