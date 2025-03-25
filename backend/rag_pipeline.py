from typing import Literal, Optional
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import chromadb
import os
import openai
import numpy as np
import logging as logger

from chunker import character_chunk_text, recursive_chunker, kamradt_chunker
from vector_store import add_chunks_to_collection, retrieve_relevant_chunks


class RAGPipeline:
    PERSISTENT_PATH = "chroma_db"

    def __init__(
        self,
        pdf_id: str,
        text: str,
        vector_store: Literal["chroma", "pinecone", "naive", "nvidia"],
        chunking_strategy: Literal["recursive", "kamradt"],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        year: Optional[str] = None,
        quarter: Optional[str] = None,
    ):
        self.pdf_id = pdf_id
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy
        self.text = text
        self.chunks = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.year = year
        self.quarter = quarter

    def process(self):
        match self.chunking_strategy:
            case "recursive":
                if self.chunk_size is not None and self.chunk_overlap is not None:
                    chunks = recursive_chunker(
                        self.text, self.chunk_size, self.chunk_overlap
                    )
                else:
                    chunks = recursive_chunker(self.text)
            case "kamradt":
                chunks = kamradt_chunker(self.text)
            case "fixed":
                chunks = character_chunk_text(self.text)
            case _:
                raise ValueError(
                    f"Unsupported chunking strategy: {self.chunking_strategy}"
                )

        match self.vector_store:
            case "chroma":
                client = chromadb.PersistentClient(path=self.PERSISTENT_PATH)
                collection = client.get_or_create_collection(self.pdf_id)
                add_chunks_to_collection(collection, chunks)
            case "pinecone":  # 384, 50
                if not os.environ.get("PINECONE_API_KEY"):
                    raise ValueError(
                        "PINECONE_API_KEY environment variable is required for Pinecone."
                    )
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedding_model.encode(chunks)

                index_name = self.pdf_id
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )

                index = pc.Index(index_name)
                vectors = [
                    {
                        "id": f"chunk-{i}",
                        "values": emb.tolist(),
                        "metadata": {"text": chunk},
                    }
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ]
                index.upsert(vectors=vectors)
            case "nvidia":
                if not os.environ.get("PINECONE_API_KEY"):
                    raise ValueError(
                        "PINECONE_API_KEY environment variable is required for Pinecone."
                    )
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedding_model.encode(chunks)

                index_name = "document-embeddings-v2"
                if index_name not in pc.list_indexes().names():
                    logger.info("NVIDIA index not found")
                    raise ValueError("NVIDIA index not found")

                index = pc.Index(index_name)
                vectors = [
                    {
                        "id": f"chunk-{i}",
                        "values": emb.tolist(),
                        "metadata": {
                            "text": chunk,
                            "year": self.year,
                            "quarter": self.quarter,
                        },
                    }
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ]
                index.upsert(vectors=vectors)
            case "naive":
                self.chunks = chunks
            case _:
                raise ValueError(f"Unsupported vector store: {self.vector_store}")

    def get_relevant_chunks(self, query: str, k: int = 5):
        match self.vector_store:
            case "chroma":
                client = chromadb.PersistentClient(path=self.PERSISTENT_PATH)
                collection = client.get_or_create_collection(self.pdf_id)
                return retrieve_relevant_chunks(collection, query, k)
            case "naive":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError(
                        "OPENAI_API_KEY environment variable is required for OpenAI."
                    )
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                query_embedding_response = client.embeddings.create(
                    input=[query], model="text-embedding-3-small"
                )
                query_embedding = query_embedding_response.data[0].embedding

                chunk_embeddings_response = client.embeddings.create(
                    input=self.chunks, model="text-embedding-3-small"
                )
                chunk_embeddings = [
                    item.embedding for item in chunk_embeddings_response.data
                ]

                similarities = [
                    (chunk, self.cosine_similarity(query_embedding, emb))
                    for chunk, emb in zip(self.chunks, chunk_embeddings)
                ]

                similarities.sort(key=lambda x: x[1], reverse=True)
                return [chunk for chunk, _ in similarities[:k]]
            case "pinecone":
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                if not os.environ.get("PINECONE_API_KEY"):
                    raise ValueError(
                        "PINECONE_API_KEY environment variable is required for Pinecone."
                    )
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                index = pc.Index(self.pdf_id)
                query_embedding = embedding_model.encode([query])[0].tolist()
                results = index.query(
                    vector=query_embedding, top_k=k, include_metadata=True
                )
                return [match["metadata"]["text"] for match in results["matches"]]
            case "nvidia":
                filter_dict = {}
                if self.year:
                    filter_dict["year"] = {"$eq": self.year}
                if self.quarter:
                    filter_dict["quarter"] = {"$eq": self.quarter}
                if not os.environ.get("PINECONE_API_KEY"):
                    raise ValueError(
                        "PINECONE_API_KEY environment variable is required for Pinecone."
                    )
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                index = pc.Index("document-embeddings-v2")
                query_embedding = embedding_model.encode([query])[0].tolist()

                results = index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None,
                )
                return [match["metadata"]["text"] for match in results["matches"]]
            case _:
                raise ValueError(f"Unsupported vector store: {self.vector_store}")

    def cosine_similarity(self, vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)
