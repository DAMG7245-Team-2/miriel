import redis.asyncio as redis
import json
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

# Constants
DEFAULT_TIMEOUT = 30  # seconds
PDF_CONTENT_PREFIX = "pdf:content:"
STREAM_CHECKPOINT_PREFIX = "stream:checkpoint:"


async def send_to_redis_stream(
    stream_name: str, message: Dict[str, Union[bytes, Any]], maxlen: int = 1000
) -> str:
    """Send a message to a Redis Stream asynchronously.

    Args:
        stream_name: Name of the Redis stream
        message: Dictionary containing the message data. Non-bytes values will be automatically
                JSON serialized to strings. Bytes values will be preserved as-is.
        maxlen: Maximum number of messages to keep in the stream

    Returns:
        str: Message ID from Redis
    """
    try:
        # Process message values - serialize all non-bytes values to JSON
        processed_message = {}
        for k, v in message.items():
            if isinstance(v, bytes):
                processed_message[k] = v
            else:
                # Serialize all non-bytes values to JSON strings
                try:
                    processed_message[k] = json.dumps(v)
                except (TypeError, ValueError):
                    # Handle non-JSON serializable objects
                    processed_message[k] = str(v)

        logger.info("Sent message to stream")
        return await redis_client.xadd(
            stream_name, processed_message, maxlen=maxlen, approximate=True
        )
    except redis.RedisError as e:
        logger.error(f"Error sending message to stream {stream_name}: {e}")
        raise


async def get_stream_checkpoint(stream_name: str) -> str:
    """Get the last processed message ID for a stream.

    Args:
        stream_name: Name of the Redis stream

    Returns:
        str: Last processed message ID or '0' if not found
    """
    try:
        checkpoint = await redis_client.get(f"{STREAM_CHECKPOINT_PREFIX}{stream_name}")
        return checkpoint if checkpoint else "0"
    except redis.RedisError as e:
        logger.error(f"Error getting stream checkpoint: {e}")
        return "0"


async def set_stream_checkpoint(stream_name: str, message_id: str) -> None:
    """Save the last processed message ID for a stream.

    Args:
        stream_name: Name of the Redis stream
        message_id: Message ID to save as checkpoint
    """
    try:
        await redis_client.set(f"{STREAM_CHECKPOINT_PREFIX}{stream_name}", message_id)
    except redis.RedisError as e:
        logger.error(f"Error setting stream checkpoint: {e}")


async def receive_from_redis_stream(
    stream_name: str, last_id: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT
) -> list:
    """Receive a message from a Redis Stream with timeout asynchronously.

    Args:
        stream_name: Name of the Redis stream
        last_id: Last message ID received, if None will use checkpoint
        timeout: Time to wait for new messages in seconds

    Returns:
        list: List of messages from the stream
    """
    try:
        if last_id is None:
            last_id = await get_stream_checkpoint(stream_name)
        logger.info(
            f"Receiving message from stream {stream_name} starting from {last_id}"
        )
        return await redis_client.xread(
            {stream_name: last_id}, count=1, block=timeout * 1000
        )
    except redis.RedisError as e:
        logger.error(f"Error receiving message from stream {stream_name}: {e}")
        raise


async def get_pdf_content_from_redis(pdf_id: str) -> Optional[bytes]:
    """Get PDF content from Redis.

    Args:
        pdf_id: Unique identifier for the PDF

    Returns:
        bytes: PDF content if found, None otherwise
    """
    try:
        logger.info(f"Getting PDF content for ID: {pdf_id}")
        # content = await redis_client.get(f"{PDF_CONTENT_PREFIX}{pdf_id}")
        stream_entries = await redis_client.xrevrange("pdf_content")
        latest_match = None
        for entry in stream_entries:
            if entry[1].get("pdf_id") == pdf_id:
                latest_match = entry
                break
        if latest_match:
            latest_match = latest_match[1].get("content")
        if latest_match is None:
            logger.error(f"PDF content not found for ID: {pdf_id}")
            raise ValueError(f"PDF content not found for ID: {pdf_id}")
        return latest_match
    except redis.RedisError as e:
        logger.error(f"Redis error getting PDF content: {e}")
        raise Exception(f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting PDF content: {e}")
        raise


async def send_llm_request(
    prompt: str, model_name: str, max_tokens: Optional[int] = None
) -> str:
    """Send LLM request to Redis Stream.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the LLM model to use
        max_tokens: Optional maximum number of tokens for the response

    Returns:
        str: Message ID from Redis
    """
    message = {
        "prompt": prompt,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }
    if max_tokens is not None:
        message["max_tokens"] = max_tokens

    return await send_to_redis_stream("llm_requests", message)


async def receive_llm_response(
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Receive LLM response from Redis Stream with timeout.

    Args:
        timeout: Time to wait for response in seconds

    Returns:
        Optional[Dict[str, Any]]: Response dictionary if received, None if timeout
    """
    try:
        # Use '$' as last_id to only get new messages
        response = await receive_from_redis_stream(
            "llm_responses", last_id="$", timeout=timeout
        )
        # response = await receive_from_redis_stream("llm_responses", timeout=timeout)
        if response:
            logger.info("Received response from Redis")
            _, messages = response[0]
            # Get the message data and deserialize from JSON
            message_data = messages[0][1]
            deserialized_message = {}
            for k, v in message_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                try:
                    # First attempt to parse as JSON
                    parsed = json.loads(value)
                    # If parsed is still a string, try parsing again
                    if isinstance(parsed, str):
                        try:
                            deserialized_message[key] = json.loads(parsed)
                        except json.JSONDecodeError:
                            deserialized_message[key] = parsed
                    else:
                        deserialized_message[key] = parsed
                except json.JSONDecodeError:
                    deserialized_message[key] = value
            return deserialized_message
        else:
            logger.info("No response received from Redis")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding LLM response: {e}")
        return None
