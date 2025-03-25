from pathlib import Path
from chunking_evaluation.chunking import RecursiveTokenChunker, KamradtModifiedChunker


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50


def recursive_chunker(
    markdown_text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    text_splitter = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs = text_splitter.split_text(markdown_text)
    return docs


def kamradt_chunker(
    markdown_text: str, avg_chunk_size: int = 300, min_chunk_size: int = 50
) -> list[str]:
    text_splitter = KamradtModifiedChunker(
        avg_chunk_size=avg_chunk_size, min_chunk_size=min_chunk_size
    )
    docs = text_splitter.split_text(markdown_text)
    return docs


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


if __name__ == "__main__":
    text = ""
    input_file = Path(
        "backend/temp_processing/output/0e34658f-6083-4447-ba10-ea5ed9d9084c.md"
    )
    with open(input_file, "r") as f:
        text = f.read()

    chunks = recursive_chunker(text)
    print(len(chunks))
    print(chunks[0])
