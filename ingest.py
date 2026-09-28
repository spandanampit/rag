from pathlib import Path

import chromadb
import ollama
import pymupdf


EMBED_MODEL = "nomic-embed-text"
SOURCE_DIRS = [Path("data"), Path("documents")]


def read_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_documents():
    documents = []

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for path in sorted(source_dir.glob("*")):
            if path.suffix.lower() == ".pdf":
                text = read_pdf(path)
            elif path.suffix.lower() == ".txt":
                text = read_txt(path)
            else:
                continue

            if text.strip():
                documents.append((path.name, text))

    return documents


documents = load_documents()

print(f"Loaded {len(documents)} document(s)")


def chunk_text(text, chunk_size=200):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def recursive_split(text, chunk_size=800, overlap=100):
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(text, separators):
        if len(text.split()) <= chunk_size:
            return [text.strip()]

        if not separators:
            words = text.split()

            chunks = []

            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

            return chunks

        separator = separators[0]

        parts = text.split(separator)

        chunks = []
        current = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            candidate = f"{current}{separator}{part}".strip()

            if len(candidate.split()) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(
                        split_text(current, separators[1:])
                    )

                current = part

        if current:
            chunks.extend(
                split_text(current, separators[1:])
            )

        return chunks

    chunks = split_text(text, separators)

    # Add overlap
    final_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        previous_words = chunks[i - 1].split()
        overlap_text = " ".join(previous_words[-overlap:])

        final_chunks.append(
            overlap_text + " " + chunk
        )

    return final_chunks



all_chunks = []
all_ids = []
all_metadatas = []

for source_name, text in documents:
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{source_name}-chunk-{i}")
        all_metadatas.append({"source": source_name, "chunk_index": i})

print(f"Created {len(all_chunks)} chunks")



BATCH_SIZE = 64

embeddings = []

for start in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[start:start + BATCH_SIZE]

    response = ollama.embed(
        model=EMBED_MODEL,
        input=batch
    )

    embeddings.extend(response["embeddings"])

    print(f"Embedded {len(embeddings)}/{len(all_chunks)} chunks", end="\r")

print()
print(f"Created {len(embeddings)} embeddings")
if embeddings:
    print(f"Embedding dimensions: {len(embeddings[0])}")



client = chromadb.PersistentClient(
    path="./chroma_db"
)


try:
    client.delete_collection(name="company_policy")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="company_policy"
)


for start in range(0, len(all_chunks), BATCH_SIZE):
    end = start + BATCH_SIZE

    collection.add(
        ids=all_ids[start:end],
        documents=all_chunks[start:end],
        embeddings=embeddings[start:end],
        metadatas=all_metadatas[start:end]
    )

print(f"Stored {len(all_chunks)} chunks successfully!")
