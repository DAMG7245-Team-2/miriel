import uuid
import chromadb
import chromadb.types
from chromadb.api.types import IncludeEnum
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_store")


def add_chunks_to_collection(collection: chromadb.Collection, chunks: list[str]) -> int:
    """
    Add document chunks to the ChromaDB collection.

    Args:
        chunks (list): List of text chunks
        url (str): Source URL of the document

    Returns:
        int: Number of chunks added
    """
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)


def retrieve_relevant_chunks(
    collection: chromadb.Collection, query: str, top_k: int = 5
):
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query (str): The query text

    Returns:
        list: List of relevant document chunks
        list: List of source URLs for each chunk
    """

    # Query the collection for similar chunks
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=[
            IncludeEnum.documents,
            IncludeEnum.metadatas,
            IncludeEnum.distances,
        ],
    )

    chunks = results["documents"][0]  # First list is for the first query

    return chunks
