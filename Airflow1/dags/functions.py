import os
import re
from pathlib import Path

import boto3
from dotenv import load_dotenv
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- Initialization ----------


from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# ---------- S3 Functions ----------
def list_all_pdfs(bucket):
    log.info("in list all pdfs message")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)
    pdf_keys = [obj["Key"] for page in page_iterator for obj in page.get("Contents", []) if obj["Key"].endswith(".pdf")]
    return pdf_keys

def download_pdf(s3_key, local_dir):
    s3 = boto3.client("s3")
    file_name = os.path.basename(s3_key)
    local_path = Path(local_dir) / file_name
    os.makedirs(local_dir, exist_ok=True)
    s3.download_file(Bucket=os.environ["S3_BUCKET"], Key=s3_key, Filename=str(local_path))
    return local_path


# ---------- Mistral OCR ----------
def extract_markdown_from_pdf(pdf_path):
    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    with open(pdf_path, "rb") as f:
        uploaded_pdf = mistral_client.files.upload(
            file={
                "file_name": pdf_path.name,
                "content": f,
            },
            purpose="ocr",
        )

        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
        ocr_result = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=False,
        )

    markdown_content = "\n".join(
        [f"### Page {i+1}\n{page.markdown}" for i, page in enumerate(ocr_result.pages)]
    )

    return markdown_content

# ---------- Chunking ----------
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return chunker.split_text(text)

# ---------- Embedding ----------
def embed_chunks(chunks):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model.encode(chunks).tolist()

# ---------- Pinecone Upload ----------
def upload_chunks_to_pinecone(chunks,embed, index_name):

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)
    
    for pdf_filename in chunks:

        chunk_list = chunks[pdf_filename]
        embed_list = embed.get(pdf_filename)

        # Skip if no embeddings (edge case)
        if not embed_list or len(chunk_list) != len(embed_list):
            print(f" Skipping {pdf_filename}: mismatched or missing embeddings")
            continue

        # Extract year and quarter
        match = re.match(r"(\d{4})-Q([1-4])\.pdf", pdf_filename)
        year = match.group(1)
        quarter = f"Q{match.group(2)}"
        if not year or not quarter:
            print(f" Invalid filename format: {pdf_filename}")
            continue

        # Prepare Pinecone payload
        chunk_prefix = f"{year}_{quarter}"
        vectors = [
            {
                "id": f"{chunk_prefix}_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "year": str(year),
                    "quarter": str(quarter),
                },
            }
            for i, (chunk, embedding) in enumerate(zip(chunk_list, embed_list))
        ]

        index.upsert(vectors=vectors)
        print(f"Uploaded {len(vectors)} chunks from {pdf_filename}")


