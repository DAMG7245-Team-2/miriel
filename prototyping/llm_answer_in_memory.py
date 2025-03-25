import os
import sys
import time
from litellm import completion
import numpy as np
import openai

from chunking_evaluation.chunking import KamradtModifiedChunker
from backend.pipelines import get_pdf_content

# In-memory storage for embeddings
embedding_store = {}


def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def retrieve_relevant_chunks(query, text, top_k=5):
    kamradt_chunker = KamradtModifiedChunker(avg_chunk_size=300, min_chunk_size=50)
    kamradt_chunks = kamradt_chunker.split_text(text)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Generate query embedding using OpenAI
    query_embedding_response = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    query_embedding = query_embedding_response.data[0].embedding

    # Generate chunk embeddings using OpenAI
    chunk_embeddings_response = client.embeddings.create(
        input=kamradt_chunks, model="text-embedding-3-small"
    )
    chunk_embeddings = [item.embedding for item in chunk_embeddings_response.data]

    similarities = [
        (chunk, cosine_similarity(query_embedding, emb))
        for chunk, emb in zip(kamradt_chunks, chunk_embeddings)
    ]

    similarities.sort(key=lambda x: x[1], reverse=True)
    return (
        "\n\n".join([chunk for chunk, _ in similarities[:top_k]]),
        similarities[:top_k],
    )


def answer_question_rag(query, model_name, text):
    # Model configuration remains identical to pinecone version
    model_mappings = {
        "gpt-4o": "openai/gpt-4o",
        "gemini-flash": "gemini/gemini-2.0-flash",
        "deepseek-chat": "deepseek/deepseek-chat",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "grok": "xai/grok-1",
    }

    if model_name not in model_mappings:
        return {
            "error": f"Invalid model. Choose from: {', '.join(model_mappings.keys())}"
        }

    api_keys = {
        "gpt-4o": os.getenv("OPENAI_API_KEY"),
        "gemini-flash": os.getenv("GEMINI_API_KEY"),
        "deepseek-chat": os.getenv("DEEPSEEK_API_KEY"),
        "claude-3-haiku": os.getenv("ANTHROPIC_API_KEY"),
        "grok": os.getenv("X_API_KEY"),
    }
    api_key = api_keys[model_name]

    if not api_key:
        return {"error": f"API key for {model_name} is missing."}

    retrieved_context, similarity_data = retrieve_relevant_chunks(query, text)

    response = completion(
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
        "answer": response.choices[0].message.content,
        "input_tokens": response["usage"]["prompt_tokens"],
        "output_tokens": response["usage"]["completion_tokens"],
        # "cost": completion_cost(completion_response=response, model='gemini/gemini-2.0-flash"'),
        "similarities": similarity_data,
    }


# Example usage (identical to pinecone version)
if __name__ == "__main__":
    pdf_id = "6fbb8546-2584-491f-84ad-ac8e0fc14ffe"
    text = get_pdf_content(pdf_id)

    if isinstance(text, str) and text.startswith("Error"):
        print(text)
        sys.exit(1)

    query = "What is the document about?"
    models_to_test = ["gemini-flash"]

    for model_name in models_to_test:
        print(f"\n{'=' * 40}\nTesting {model_name.upper()} Model\n{'=' * 40}")
        start_time = time.time()

        try:
            result = answer_question_rag(query, model_name, text)
            elapsed_time = time.time() - start_time

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            print("\nRetrieved Context Chunks:")
            for idx, (chunk, score) in enumerate(result["similarities"], 1):
                print(f"Chunk {idx} [Similarity: {score:.4f}]: {chunk[:100]}...")

            print("\nQuestion:", result.get("question", "N/A"))
            print("\nAnswer:", result.get("answer", "N/A"))
            print("\nPerformance Metrics:")
            print(f"Time Taken: {elapsed_time:.2f}s")
            print(f"Input Tokens: {result.get('input_tokens', 0)}")
            print(f"Output Tokens: {result.get('output_tokens', 0)}")
            print(f"Cost: ${result.get('cost', 0):.6f}")

        except Exception as e:
            print(f"Error answering question with {model_name}: {str(e)}")
