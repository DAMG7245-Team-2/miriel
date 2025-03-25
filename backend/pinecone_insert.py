import os
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count
from pinecone import Pinecone, ServerlessSpec


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def recursive_chunk_document(text, chunk_size=384, chunk_overlap=50):
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return chunker.split_text(text)


def character_chunk_text(text, chunk_size=384, overlap=50):
    """
    Split text into chunks based on character count.

    Args:
        text: Text to split
        chunk_size: Maximum number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    stride = chunk_size - overlap
    current_idx = 0

    while current_idx < len(text):
        # Take chunk_size characters starting from current_idx
        chunk = text[current_idx : current_idx + chunk_size]
        if not chunk:  # Break if we're out of text
            break
        chunks.append(chunk)
        current_idx += stride  # Move forward by stride

    return chunks


def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(model_name)
    return embedding_model.encode(texts)


# Load and process file
file_path = "nvidia_2024.md"
content = read_markdown_file(file_path)
chunks = recursive_chunk_document(
    content
)  # character_chunk_text(content) --Change this for the type of chunking

# Create embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_texts(chunks)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "document-embeddings"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Prepare and upsert vectors
vectors = [
    {"id": f"chunk-{i}", "values": embedding.tolist(), "metadata": {"text": chunk}}
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

index.upsert(vectors=vectors)

print(f"Loaded {len(chunks)} chunks into Pinecone index '{index_name}'.")
