import os
import litellm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Pinecone setup
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("document-embeddings")
index_nvidia = pc.Index("document-embeddings-v2")
# Model and API key setup
api_keys = {
    "gpt-4o": os.getenv("OPENAI_API_KEY"),
    "gemini-flash": os.getenv("GOOGLE_API_KEY"),
    "deepseek-chat": os.getenv("DEEPSEEK_API_KEY"),
    "claude-3-haiku": os.getenv("ANTHROPIC_API_KEY"),
    "grok": os.getenv("X_API_KEY"),
}

model_mappings = {
    "gpt-4o": "openai/gpt-4o",
    "gemini-flash": "gemini/gemini-2.0-flash",
    "deepseek-chat": "deepseek/deepseek-chat",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "grok": "xai/grok-1",
}


def retrieve_relevant_chunks(query, embedding_model, top_k=5):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])


def answer_question_rag(query, model_name, embedding_model):
    if model_name not in model_mappings:
        return {
            "error": f"Invalid model. Choose from: {', '.join(model_mappings.keys())}"
        }
    api_key = api_keys[model_name]
    if not api_key:
        return {"error": f"API key for {model_name} is missing."}

    retrieved_context = retrieve_relevant_chunks(query, embedding_model)

    response = litellm.completion(
        model=model_mappings[model_name],
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant. Answer the user's question based on the following retrieved information.",
            },
            {
                "role": "user",
                "content": f"Context:\n{retrieved_context}\n\nQuestion: {query}",
            },
        ],
        api_key=api_key,
        stream=False,
    )

    return {
        "question": query,
        "answer": response["choices"][0]["message"]["content"],
        "input_tokens": response["usage"]["prompt_tokens"],
        "output_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
    }


def retrieve_relevant_chunks_nvidia(
    query, embedding_model, top_k=5, year=None, quarter=None
):
    query_embedding = embedding_model.encode([query])[0].tolist()

    # Build filter dictionary
    filter_dict = {}
    if year:
        filter_dict["year"] = {"$eq": year}
    if quarter:
        filter_dict["quarter"] = {"$eq": quarter}

    results = index_nvidia.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n".join(chunks)


def answer_question_rag_nvidia(
    query, model_name, embedding_model, year=None, quarter=None
):
    if model_name not in model_mappings:
        return {
            "error": "Invalid model selection. Choose from: "
            + ", ".join(model_mappings.keys())
        }

    model = model_mappings[model_name]
    api_key = api_keys[model_name]

    if not api_key:
        return {"error": f"API key for {model_name} is missing."}

    try:
        # Retrieve chunks with optional year & quarter filter
        retrieved_context = retrieve_relevant_chunks_nvidia(
            query, embedding_model, year=year, quarter=quarter
        )

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant. Answer the user's question based on the context given to you.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{retrieved_context}\n\nQuestion: {query}",
                },
            ],
            api_key=api_key,
            stream=False,
        )

        return {
            "question": query,
            "answer": response["choices"][0]["message"]["content"],
            "input_tokens": response["usage"]["prompt_tokens"],
            "output_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Did they talk about risk in this document?"
    selected_model = "gpt-4o"

    result = answer_question_rag(query, selected_model, embedding_model)

    print("\nQuestion:", result.get("question"))
    print("\nAnswer:", result.get("answer"))
    print("\nTotal Tokens:", result.get("total_tokens"))
    print("\nInput Tokens:", result.get("input_tokens"))
    print("\nOutput Tokens:", result.get("output_tokens"))

    result_nvidia = answer_question_rag_nvidia(
        query, selected_model, embedding_model, year="2024", quarter="Q4"
    )

    print("\nQuestion:", result_nvidia.get("question"))
    print("\nAnswer:", result_nvidia.get("answer"))
    print("\nTotal Tokens:", result_nvidia.get("total_tokens"))
    print("\nInput Tokens:", result_nvidia.get("input_tokens"))
    print("\nOutput Tokens:", result_nvidia.get("output_tokens"))
