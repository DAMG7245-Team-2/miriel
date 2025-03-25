import asyncio
import json
import logging
from typing import Dict, Any, Optional

from llm_manager import LLMManager
from redis_manager import (
    receive_from_redis_stream,
    send_to_redis_stream,
    get_pdf_content_from_redis,
    set_stream_checkpoint,
)

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.request_handlers = {
            "question": self._handle_question,
            "summary": self._handle_summary,
        }

    async def _handle_question(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = request.get("question")
            pdf_id = request.get("pdf_id")
            max_tokens = request.get("max_tokens", 5000)
            model = request.get("model", "gemini/gemini-2.0-flash")

            if not question or not pdf_id:
                raise ValueError("Question and PDF ID are required")

            try:
                # Get PDF content from Redis with improved error handling
                pdf_content = await get_pdf_content_from_redis(pdf_id)
                logger.info(f"PDF content length: {len(pdf_content)}")
                context = pdf_content
                # .decode("utf-8")

                answer, usage_metrics = await self.llm_manager.ask_question(
                    context=context,
                    question=question,
                    max_tokens=max_tokens,
                    model=model,
                )

                return {
                    "type": "question_response",
                    "content": answer,
                    "status": "success",
                    "usage": usage_metrics,
                }
            except ValueError as e:
                logger.error(f"PDF content error: {str(e)}")
                return {
                    "type": "question_response",
                    "content": f"Error: {str(e)}",
                    "status": "error",
                }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {"type": "question_response", "content": str(e), "status": "error"}

    async def _handle_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pdf_id = request.get("pdf_id")
            max_tokens = request.get("max_tokens", 5000)
            model = request.get("model", "gemini/gemini-2.0-flash")

            if not pdf_id:
                raise ValueError("PDF ID is required")

            # Get PDF content from Redis
            pdf_content = await get_pdf_content_from_redis(pdf_id)
            if not pdf_content:
                raise ValueError("PDF content not found")

            # Decode PDF content from bytes
            text = pdf_content
            # .decode("utf-8")

            summary, usage_metrics = await self.llm_manager.get_summary(
                text=text, max_tokens=max_tokens, model=model
            )

            return {
                "type": "summary_response",
                "content": summary,
                "status": "success",
                "usage": usage_metrics,
            }
        except Exception as e:
            logger.error(f"Error processing summary: {str(e)}")
            return {"type": "summary_response", "content": str(e), "status": "error"}

    async def process_request(
        self, request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            request_type = request.get("type")
            handler = self.request_handlers.get(request_type)

            if not handler:
                logger.error(f"Unknown request type: {request_type}")
                return {
                    "type": "error",
                    "content": f"Unknown request type: {request_type}",
                    "status": "error",
                }

            return await handler(request)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {"type": "error", "content": str(e), "status": "error"}

    async def start(self):
        logger.info("Starting LLM Service...")
        max_retries = 3

        while True:
            try:
                # Read from llm_requests stream using checkpoint mechanism
                messages = await receive_from_redis_stream("llm_requests")

                if not messages:
                    await asyncio.sleep(0.1)  # Avoid busy waiting
                    continue

                stream_name, stream_messages = messages[0]
                message_id, message_data = stream_messages[0]

                # Update checkpoint with latest processed message ID
                await set_stream_checkpoint("llm_requests", message_id)

                # Process request
                try:
                    # Decode and parse JSON values from Redis stream
                    request = {}
                    for k, v in message_data.items():
                        key = k
                        # .decode()
                        try:
                            # Try to parse JSON value
                            value = json.loads(
                                v
                                #    .decode()
                            )
                        except json.JSONDecodeError:
                            # If not JSON, use raw decoded value
                            value = v
                            # .decode()
                        request[key] = value

                    if "type" not in request:
                        logger.error("Request type is missing")
                        continue

                    response = await self.process_request(request)
                except Exception as e:
                    logger.error(f"Error decoding request data: {str(e)}")
                    continue

                if response:
                    # Send response to llm_responses stream with retries
                    retries = 0
                    while retries < max_retries:
                        try:
                            await send_to_redis_stream("llm_responses", response)
                            break
                        except Exception as e:
                            retries += 1
                            if retries == max_retries:
                                logger.error(
                                    f"Failed to send response after {max_retries} attempts: {str(e)}"
                                )
                            else:
                                logger.warning(
                                    f"Error sending response (attempt {retries}): {str(e)}"
                                )
                                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in service loop: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying


async def run_service():
    service = LLMService()
    await service.start()


def main():
    asyncio.run(run_service())


if __name__ == "__main__":
    main()
