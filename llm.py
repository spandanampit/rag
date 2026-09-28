import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Explain what RAG is in one sentence."
        }
    ]
)


print(response["message"]["content"])