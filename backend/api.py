import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging

from cloud_ops import download_file_from_s3, list_objects_in_s3
from llm_manager import LLMManager
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

from redis_manager import send_to_redis_stream, receive_llm_response
from pipelines import (
    store_uploaded_pdf,
    get_pdf_content,
)


load_dotenv()
app = FastAPI()

# LLM Service instance
llm_service = None

bucket_name = "miriel"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_service
    from llm_service import LLMService

    llm_service = LLMService()
    asyncio.create_task(llm_service.start())
    logger.info("LLM Service started")

    yield

    # Shutdown
    logger.info("Shutting down LLM Service")


app = FastAPI(lifespan=lifespan)


class URLRequest(BaseModel):
    url: str


class PDFSelection(BaseModel):
    pdf_id: str


class PDFUploadResponse(BaseModel):
    pdf_id: str
    status: str
    message: Optional[str] = None


class SummaryRequest(BaseModel):
    pdf_id: str = Field(..., min_length=8, max_length=100)
    summary_length: int = Field(200, gt=50, lt=1000)
    max_tokens: int = Field(500, gt=100, lt=2000)
    model: str = Field(
        "gemini/gemini-2.0-flash", description="LLM model to use for summary generation"
    )


class QuestionRequest(BaseModel):
    pdf_id: str
    question: str = Field(..., min_length=10)
    max_tokens: int = Field(500, gt=100, lt=2000)
    model: str = Field(
        "gemini/gemini-2.0-flash", description="LLM model to use for question answering"
    )


active_rag_pipelines = {}


@app.get("/list_uploaded_pdfs", tags=["Assignment 4.1"])
async def list_uploaded_pdfs():
    return list_objects_in_s3(bucket_name, "pdfs/raw")


@app.post("/select_uploaded_pdf", tags=["Assignment 4.1"])
async def select_uploaded_pdf(
    request: PDFSelection,
    parser: Literal["docling", "mistral"],
    chunking_strategy: Literal["recursive", "kamradt", "fixed"],
    vector_store: Literal["chroma", "pinecone", "naive"],
):
    path = download_file_from_s3(
        f"pdfs/raw/{request.pdf_id}.pdf", "temp_processing/output", bucket_name
    )
    if path is None:
        logger.error(f"Failed to select PDF {request.pdf_id}")
        raise HTTPException(status_code=404, detail="Failed to select PDF")

    upload_file = UploadFile(
        file=open(path, "rb"),
        filename=f"{request.pdf_id}.pdf",
        headers={"content-type": "application/pdf"},
    )

    return await upload_pdf(upload_file, parser, chunking_strategy, vector_store)


@app.post("/upload_pdf", status_code=status.HTTP_201_CREATED, tags=["Assignment 4.1"])
async def upload_pdf(
    file: UploadFile,
    parser: Literal["docling", "mistral"],
    chunking_strategy: Literal["recursive", "kamradt", "fixed"],
    vector_store: Literal["chroma", "pinecone", "naive"],
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        pdf_id = store_uploaded_pdf(contents, parser)
        contents = get_pdf_content(pdf_id)
        rag_pipeline = RAGPipeline(
            pdf_id=pdf_id,
            text=contents,
            chunking_strategy=chunking_strategy,
            vector_store=vector_store,
        )
        rag_pipeline.process()
        logger.info(f"PDF {pdf_id} processed")
        active_rag_pipelines[pdf_id] = rag_pipeline

        return PDFUploadResponse(
            pdf_id=pdf_id, status="success", message=f"PDF stored with ID: {pdf_id}"
        )
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.post("/summarize/", response_model=dict, tags=["Assignment 4.1"])
async def generate_summary(request: SummaryRequest):
    # logger = logging.getLogger(__name__)

    # Send request to Redis stream and wait for response
    if request.pdf_id not in active_rag_pipelines:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        content = get_pdf_content(request.pdf_id)
        logger.info(
            f"Sending PDF content to Redis stream for {request.pdf_id} content: {content}"
        )
        await send_to_redis_stream(
            "pdf_content",
            {
                "pdf_id": request.pdf_id,
                "content": content,
            },
        )
        logger.info(f"PDF content sent to Redis stream for {request.pdf_id}")
        await send_to_redis_stream(
            "llm_requests",
            {
                "type": "summary",
                "pdf_id": request.pdf_id,
                "max_tokens": request.max_tokens,
                "model": request.model,
            },
        )
        logger.info(f"Summary request sent to Redis stream for {request.pdf_id}")
        response = await receive_llm_response()
        logger.info(f"Summary response received from Redis stream for {request.pdf_id}")
        if not response:
            raise HTTPException(status_code=408, detail="LLM response timeout")

        # Extract usage metrics from response
        usage_metrics = response.get("usage", {})
        if not usage_metrics:
            logger.warning("No usage metrics found in response")

        return {
            "summary": response.get("content", ""),
            "usage_metrics": {
                "input_tokens": usage_metrics.get("input_tokens", 0),
                "output_tokens": usage_metrics.get("output_tokens", 0),
                "total_tokens": usage_metrics.get("total_tokens", 0),
                "cost": usage_metrics.get("cost", 0.0),
            },
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Summary generation failed")


@app.post("/ask_question", status_code=status.HTTP_200_OK, tags=["Assignment 4.1"])
async def answer_pdf_question(request: QuestionRequest):
    # logger = logging.getLogger(__name__)
    if request.pdf_id not in active_rag_pipelines:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        rag_pipeline: RAGPipeline = active_rag_pipelines[request.pdf_id]
        relevant_chunks = rag_pipeline.get_relevant_chunks(request.question, 5)
        logger.info(f"PDF {request.pdf_id} retrieved {len(relevant_chunks)} chunks")
        context = "\n\n".join(relevant_chunks)
        logger.info(f"PDF {request.pdf_id} context: {context}")
        await send_to_redis_stream(
            "pdf_content",
            {
                "pdf_id": request.pdf_id.encode("utf-8"),
                "content": context.encode("utf-8"),
            },
        )
        logger.info(f"PDF {request.pdf_id} context sent to stream")
        await asyncio.sleep(1)
        await send_to_redis_stream(
            "llm_requests",
            {
                "type": "question",
                "pdf_id": request.pdf_id.encode("utf-8"),
                "question": request.question.encode("utf-8"),
                "max_tokens": request.max_tokens,
            },
        )
        logger.info(f"Question {request.question} sent to stream")
        response = await receive_llm_response()
        logger.info(f"Question {request.question} response: {response}")
        if not response:
            raise HTTPException(status_code=408, detail="LLM response timeout")

        # Extract content from response
        content = response.get("content")
        if not content:
            logger.error(f"Missing content in response: {response}")
            raise HTTPException(
                status_code=500, detail="Invalid response format from LLM service"
            )

        # Check response status
        if response.get("status") != "success":
            logger.error(f"Failed response status: {response}")
            raise HTTPException(status_code=500, detail="LLM service processing failed")

        # Extract usage metrics from response
        usage_metrics = response.get("usage", {})
        if not usage_metrics:
            logger.warning("No usage metrics found in response")

        return {
            "question": request.question,
            "answer": content,
            "source_pdf": request.pdf_id,
            "usage_metrics": {
                "input_tokens": usage_metrics.get("input_tokens", 0),
                "output_tokens": usage_metrics.get("output_tokens", 0),
                "total_tokens": usage_metrics.get("total_tokens", 0),
                "cost": usage_metrics.get("cost", 0.0),
            },
            "status": "success",
        }
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Question answering failed")


@app.post("/ask_nvidia", status_code=status.HTTP_200_OK, tags=["Assignment 4.2"])
async def ask_nvidia(
    request: QuestionRequest, year: Optional[str] = None, quarter: Optional[str] = None
):
    try:
        pipeline = RAGPipeline(
            pdf_id="0000",
            text="",
            vector_store="nvidia",
            chunking_strategy="recursive",
            year=year,
            quarter=quarter,
        )
        pipeline.get_relevant_chunks(query=request.question, k=5)

        context = pipeline.get_relevant_chunks(query=request.question, k=5)
        logger.info(f"----------context: {context}")

        # Define the user message with query and context
        user_message = f"""Question: {request.question}
        
        Context information:
        {context}
        
        Please answer the question based on the context information provided."""

        llm_manager = LLMManager()
        content, usage_metrics = await llm_manager.get_llm_response(
            user_message, request.model
        )

        if not usage_metrics:
            logger.warning("No usage metrics found in response")

        return {
            "question": request.question,
            "answer": content,
            "source_pdf": request.pdf_id,
            "usage_metrics": {
                "input_tokens": usage_metrics.get("input_tokens", 0),
                "output_tokens": usage_metrics.get("output_tokens", 0),
                "total_tokens": usage_metrics.get("total_tokens", 0),
                "cost": usage_metrics.get("cost", 0.0),
            },
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in NVIDIA RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
