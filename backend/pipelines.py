import os
import shutil
import logging
from typing import Literal
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

from cloud_ops import upload_file_to_s3
from python_pdf_extraction import (
    extract_text_with_docling,
    extract_text_with_mistral,
)

base_dir = Path("./temp_processing")
output = base_dir / Path("output")
s3_bucket = "miriel"
s3_pdf_input_prefix = "pdfs/raw"
s3_html_input_prefix = "html/raw"


def store_uploaded_pdf(
    pdf_content: bytes, parser: Literal["docling", "mistral"]
) -> str:
    """Store uploaded PDF and convert to markdown for LLM processing"""
    logger = logging.getLogger(__name__)
    pdf_id = str(uuid.uuid4())
    os.makedirs(output, exist_ok=True)

    # Save original PDF
    pdf_path = output / f"{pdf_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    match parser:
        case "docling":
            # Convert to markdown using existing pipeline
            markdown_output = pdf_to_md_docling(pdf_path, pdf_id)

        case "mistral":
            # Convert to markdown using existing pipeline
            markdown_output = pdf_to_md_mistral(pdf_path, pdf_id)
    markdown_path = output / f"{pdf_id}.md"

    if not markdown_output.get("markdown"):
        logger.error(f"Markdown generation failed for PDF {pdf_id}")
        raise ValueError("Failed to generate markdown content")

    shutil.move(str(markdown_output["markdown"]), str(markdown_path))
    logger.info(f"Stored PDF {pdf_id} at {markdown_path}")
    return pdf_id


def get_pdf_content(pdf_id: str) -> str:
    """Retrieve stored markdown content for LLM processing"""
    markdown_path = output / f"{pdf_id}.md"
    if not markdown_path.exists():
        raise FileNotFoundError(f"No content found for PDF ID: {pdf_id}")

    with open(markdown_path, "r") as f:
        return f.read()


def pdf_to_md_docling(file: Path, job_name: uuid):
    s3_prefix_text = "pdfs/md"

    output_data = {"markdown": None, "images": None, "tables": None}

    # Step 1: Upload input PDF to S3
    input_pdf_s3_key = f"{s3_pdf_input_prefix}/{job_name}.pdf"
    upload_file_to_s3(str(file), input_pdf_s3_key, bucket_name=s3_bucket)

    # Step 2: Extract text and upload Markdown file to S3
    markdown_local_path = output / f"{job_name}.md"
    extract_text_with_docling(file, markdown_local_path)

    if markdown_local_path.exists() and not is_file_empty(markdown_local_path):
        output_data["markdown"] = markdown_local_path
        markdown_s3_key = f"{s3_prefix_text}/{Path(markdown_local_path).name}"
        upload_file_to_s3(
            str(markdown_local_path), markdown_s3_key, bucket_name=s3_bucket
        )

    return output_data


def pdf_to_md_mistral(file: Path, job_name: uuid):
    s3_prefix_text = "pdfs/md"

    output_data = {"markdown": None, "images": None, "tables": None}

    # Step 1: Upload input PDF to S3
    input_pdf_s3_key = f"{s3_pdf_input_prefix}/{job_name}.pdf"
    upload_file_to_s3(str(file), input_pdf_s3_key, bucket_name=s3_bucket)

    # Step 2: Extract text and upload Markdown file to S3
    markdown_local_path = output / f"{job_name}.md"
    extract_text_with_mistral(file, markdown_local_path)

    if markdown_local_path.exists() and not is_file_empty(markdown_local_path):
        output_data["markdown"] = markdown_local_path
        markdown_s3_key = f"{s3_prefix_text}/{Path(markdown_local_path).name}"
        upload_file_to_s3(
            str(markdown_local_path), markdown_s3_key, bucket_name=s3_bucket
        )

    return output_data


def is_file_empty(file_path: Path) -> bool:
    return file_path.stat().st_size == 0


def get_job_name():
    return uuid.uuid4()


def clean_temp_files():
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output, exist_ok=True)
