import chromadb 

client = chromadb.PersistentClient(path="./chroma_db")

collection=client.get_or_create_collection(name="company_policy")

print("Collection created successfully!")