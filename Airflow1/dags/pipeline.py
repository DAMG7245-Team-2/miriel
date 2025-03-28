
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import re
from airflow.utils.log.logging_mixin import LoggingMixin
import requests
from bs4 import BeautifulSoup
import boto3

log = LoggingMixin().log


load_dotenv()


from functions import (
    list_all_pdfs,
    download_pdf,
    extract_markdown_from_pdf,
    chunk_text,
    embed_chunks,
    upload_chunks_to_pinecone
)


load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

HTML_PATH = os.path.join(os.path.dirname(__file__), "NVIDIA-fin.html")
DOWNLOAD_DIR = "nvidia_reports"
BUCKET_NAME = "nvidia-ak"
S3_PREFIX = ""

# === TASK 1: Download PDFs ===
def download_pdfs():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    rows = soup.find_all('div', class_=lambda c: c and 'module-financial-table_body-row' in c and 'Form 10-Q/Form 10-K' in c)
    documents = []

    for row in rows:
        links = row.find_all('a', href=True)
        for link in links:
            href = link['href']
            parts = href.split('/')
            if len(parts) >= 8:
                year = parts[6]
                quarter = parts[7].upper()
                if year.isdigit() and quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                    documents.append((int(year), quarter, href))

    quarter_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    documents.sort(key=lambda x: (-x[0], quarter_order.get(x[1], 5)))

    for year, quarter, url in documents:
        filename = f"{year}-{quarter}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        print(f"\U0001F4E5 Downloading {filename}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed: {filename} → {e}")

# === TASK 2 and 3: Delete and Upload to S3 ===
def sync_folder_to_s3():
    s3 = boto3.client('s3')

    print(f"Deleting existing files from s3://{BUCKET_NAME}/{S3_PREFIX}")
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX):
        if 'Contents' in page:
            objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
            s3.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects_to_delete})
            print(f" Deleted {len(objects_to_delete)} objects")

    print(f"Uploading files from {DOWNLOAD_DIR} to s3://{BUCKET_NAME}/{S3_PREFIX}")
    for filename in os.listdir(DOWNLOAD_DIR):
        if filename.endswith(".pdf"):
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            s3_key = f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename
            print(f"→ Uploading {filename} to {s3_key}")
            s3.upload_file(local_path, BUCKET_NAME, s3_key, ExtraArgs={'ContentType': 'application/pdf'})
    print("Upload complete.")



S3_BUCKET = "nvidia-ak"
LOCAL_TEMP_DIR = Path("/tmp/airflow_pdf")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
}

dag = DAG(
    "s3_to_pinecone_pipeline_v5.1",
    default_args=default_args,
    #schedule_interval="*/21 * * * *",
    catchup=False,
)


web_scrape = PythonOperator(
    task_id='web_scrape',
    python_callable=download_pdfs,
    dag=dag,
)

upload_s3 = PythonOperator(
    task_id='upload_s3',
    python_callable=sync_folder_to_s3,
    dag=dag,
)


def get_pdf_keys(**kwargs):
    pdf_keys = list_all_pdfs(S3_BUCKET)
    print(f"Found {len(pdf_keys)} PDFs: {pdf_keys}")
    if not pdf_keys:
        raise ValueError("No PDFs found in the S3 bucket!")
    kwargs["ti"].xcom_push(key="pdf_keys", value=pdf_keys)


def convert_to_markdown(**kwargs):
    pdf_keys = kwargs["ti"].xcom_pull(task_ids="get_from_s3", key="pdf_keys")
    markdowns = {}
    for pdf_key in pdf_keys:
        local_path = download_pdf(pdf_key, LOCAL_TEMP_DIR)
        markdown = extract_markdown_from_pdf(local_path)
        markdowns[pdf_key] = markdown
    kwargs["ti"].xcom_push(key="markdowns", value=markdowns)

def chunk_documents(**kwargs):
    markdowns = kwargs["ti"].xcom_pull(task_ids="convert_to_markdown", key="markdowns")
    all_chunks = {}
    for key, text in markdowns.items():
        chunks = chunk_text(text)
        all_chunks[key] = chunks
    kwargs["ti"].xcom_push(key="chunks", value=all_chunks)

def embed_chunks_task(**kwargs):
    all_chunks = kwargs["ti"].xcom_pull(task_ids="chunk_docs", key="chunks")
    all_embeddings = {}
    for key, chunks in all_chunks.items():
        embeddings = embed_chunks(chunks)
        all_embeddings[key] = embeddings
    kwargs["ti"].xcom_push(key="embeddings", value=all_embeddings)

def upload_to_pinecone_task(**kwargs):
    chunks = kwargs["ti"].xcom_pull(task_ids="chunk_docs", key="chunks")
    embeddings = kwargs["ti"].xcom_pull(task_ids="embed_chunks", key="embeddings")
    upload_chunks_to_pinecone(chunks,embeddings, index_name="doc-embeddings-v2")

get_from_s3 = PythonOperator(
    task_id="get_from_s3",
    python_callable=get_pdf_keys,
    provide_context=True,
    dag=dag,
)

convert_to_markdown_task = PythonOperator(
    task_id="convert_to_markdown",
    python_callable=convert_to_markdown,
    provide_context=True,
    dag=dag,
)

chunk_docs = PythonOperator(
    task_id="chunk_docs",
    python_callable=chunk_documents,
    provide_context=True,
    dag=dag,
)

embed_chunks_op = PythonOperator(
    task_id="embed_chunks",
    python_callable=embed_chunks_task,
    provide_context=True,
    dag=dag,
)

upload_to_pinecone_op = PythonOperator(
    task_id="upload_to_pinecone",
    python_callable=upload_to_pinecone_task,
    provide_context=True,
    dag=dag,
)

web_scrape >> upload_s3 >> get_from_s3 >> convert_to_markdown_task >> chunk_docs >> embed_chunks_op >> upload_to_pinecone_op

