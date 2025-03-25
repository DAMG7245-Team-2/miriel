import chromadb
from vector_store import retrieve_relevant_chunks

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./backend/chroma_db")
    collection = client.get_or_create_collection("test1")
    print(collection.count())
    # text = ""
    # with open("backend/temp_processing/output/2022-Q1.md", "r") as f:
    #     text = f.read()
    # chunks = recursive_chunker(text)
    # num_chunks = add_chunks_to_collection(collection, chunks)
    # print(f"Added {num_chunks} chunks to the collection")
    documents = retrieve_relevant_chunks(
        collection, "What are the risks listed in the document?"
    )

    print(documents)
