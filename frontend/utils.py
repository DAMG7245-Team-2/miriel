import asyncio
import aiohttp
import logging
import threading
import queue
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WS_URL = "ws://localhost:8000/chat"

# Global queues to store received messages safely between threads
received_messages = {}


class WebSocketManager:
    """Manages websocket connections for multiple chat sessions"""

    def __init__(self):
        self.connections = {}  # Store active connections by chat_id
        self.message_callbacks = {}  # Store callbacks by chat_id
        self.running = {}  # Track running status for each connection
        self.pdf_ids = {}  # Store pdf_id by chat_id
        self.loops = {}  # Store event loops by chat_id
        self.message_queues = {}  # Store message queues by chat_id

        # Thread-safe queue for responses
        self.response_queues = {}

    def connect(
        self, chat_id: str, pdf_id: str, message_callback: Callable[[str], None]
    ):
        """
        Create a websocket connection for a given chat

        Args:
            chat_id: The ID of the chat
            pdf_id: The ID of the PDF to connect to
            message_callback: Function to call when a message is received
        """
        if chat_id in self.connections:
            logger.info(f"Connection for chat {chat_id} already exists")
            return

        # Initialize thread-safe queue for this chat
        global received_messages
        received_messages[chat_id] = queue.Queue()

        self.message_callbacks[chat_id] = message_callback
        self.running[chat_id] = True
        self.pdf_ids[chat_id] = pdf_id
        self.message_queues[chat_id] = asyncio.Queue()
        self.response_queues[chat_id] = queue.Queue()

        # Start the connection in a separate thread
        thread = threading.Thread(
            target=self._run_connection_loop, args=(chat_id,), daemon=True
        )
        thread.start()
        logger.info(
            f"Started websocket connection for chat {chat_id} with PDF {pdf_id}"
        )

    def disconnect(self, chat_id: str):
        """Close the connection for a given chat ID"""
        if chat_id in self.running:
            self.running[chat_id] = False
            logger.info(f"Marked connection for chat {chat_id} to be closed")

            # Clean up response queue
            global received_messages
            if chat_id in received_messages:
                del received_messages[chat_id]

    def send_message(self, chat_id: str, message: str):
        """
        Send a message through the websocket

        Args:
            chat_id: The ID of the chat to send the message to
            message: The message to send
        """
        if chat_id not in self.running or not self.running[chat_id]:
            logger.error(f"No active connection for chat {chat_id}")
            return

        try:
            # Add message to the queue for sending
            if chat_id in self.message_queues:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.message_queues[chat_id].put(message))
                loop.close()
                logger.info(f"Queued message for chat {chat_id}")
            else:
                logger.error(f"No message queue for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error queueing message: {e}")

    def get_responses(self, chat_id: str):
        """Get any responses that have been received for a chat session"""
        global received_messages
        responses = []

        if chat_id in received_messages:
            while not received_messages[chat_id].empty():
                try:
                    responses.append(received_messages[chat_id].get_nowait())
                except queue.Empty:
                    break

        return responses

    def _run_connection_loop(self, chat_id: str):
        """Run the websocket connection loop in a separate thread"""
        # Create a new event loop for asyncio in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loops[chat_id] = loop

        try:
            loop.run_until_complete(self._maintain_connection(chat_id))
        except Exception as e:
            logger.error(f"Error in websocket loop: {e}")
        finally:
            loop.close()
            if chat_id in self.loops:
                del self.loops[chat_id]

    async def _maintain_connection(self, chat_id: str):
        """Maintain the websocket connection and reconnect if necessary"""
        pdf_id = self.pdf_ids.get(chat_id)
        if not pdf_id:
            logger.error(f"No PDF ID found for chat {chat_id}")
            return

        while self.running.get(chat_id, False):
            try:
                async with aiohttp.ClientSession() as session:
                    # Connect to the websocket with the PDF ID as a parameter
                    async with session.ws_connect(
                        WS_URL, params={"pdf_id": pdf_id}, heartbeat=30
                    ) as ws:
                        logger.info(
                            f"Connected to websocket for chat {chat_id} with PDF {pdf_id}"
                        )
                        self.connections[chat_id] = ws

                        # Create tasks for sending and receiving messages
                        send_task = asyncio.create_task(
                            self._send_messages(chat_id, ws)
                        )
                        receive_task = asyncio.create_task(
                            self._receive_messages(chat_id, ws)
                        )

                        # Wait for either task to complete (which means connection is broken or closed)
                        done, pending = await asyncio.wait(
                            [send_task, receive_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Cancel the pending task
                        for task in pending:
                            task.cancel()

                        # Remove from connections when done
                        if chat_id in self.connections:
                            del self.connections[chat_id]

                        # If we're not running anymore, clean up
                        if not self.running.get(chat_id, False):
                            self._cleanup_resources(chat_id)
                            return

                # If we get here, the connection was closed but we should reconnect
                logger.info(f"Reconnecting websocket for chat {chat_id}")
                await asyncio.sleep(2)  # Wait before reconnecting

            except Exception as e:
                logger.error(f"Connection error for chat {chat_id}: {e}")
                await asyncio.sleep(2)  # Wait before reconnecting

        # Final cleanup if the loop exits
        self._cleanup_resources(chat_id)
        logger.info(f"Websocket connection loop for chat {chat_id} ended")

    def _cleanup_resources(self, chat_id: str):
        """Clean up resources for a chat"""
        if chat_id in self.message_callbacks:
            del self.message_callbacks[chat_id]
        if chat_id in self.running:
            del self.running[chat_id]
        if chat_id in self.pdf_ids:
            del self.pdf_ids[chat_id]
        if chat_id in self.message_queues:
            del self.message_queues[chat_id]
        if chat_id in self.response_queues:
            del self.response_queues[chat_id]
        logger.info(f"Cleaned up resources for chat {chat_id}")

    async def _send_messages(self, chat_id: str, ws):
        """Send messages from the queue"""
        queue = self.message_queues.get(chat_id)
        if not queue:
            logger.error(f"No message queue for chat {chat_id}")
            return

        try:
            while self.running.get(chat_id, False):
                # Get the next message from the queue with timeout
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.5)
                    await ws.send_str(message)
                    logger.info(f"Sent message to chat {chat_id}")
                except asyncio.TimeoutError:
                    # No message available, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    return
        except asyncio.CancelledError:
            # Task was cancelled, this is expected
            logger.info(f"Send messages task cancelled for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error in send messages task: {e}")

    async def _receive_messages(self, chat_id: str, ws):
        """Receive messages and call the callback"""
        global received_messages

        try:
            while self.running.get(chat_id, False):
                try:
                    # Wait for messages
                    msg = await asyncio.wait_for(ws.receive(), timeout=60)

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Store the message in the thread-safe queue
                        if chat_id in received_messages:
                            received_messages[chat_id].put(msg.data)
                            logger.info(f"Received message for chat {chat_id}")
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        logger.warning(f"Websocket closed for chat {chat_id}: {msg}")
                        return
                except asyncio.TimeoutError:
                    # No message received within timeout, perform a ping to keep connection alive
                    await ws.ping()
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    return
        except asyncio.CancelledError:
            # Task was cancelled, this is expected
            logger.info(f"Receive messages task cancelled for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error in receive messages task: {e}")


# Create a global instance
ws_manager = WebSocketManager()


# Keep the original chat_with_pdf function for CLI usage
async def chat_with_pdf(pdf_id: str):
    """
    Connect to a websocket, send questions and receive responses in real-time.

    Args:
        pdf_id: The ID of the PDF to chat about
    """
    try:
        # Create a session and connect to the websocket
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL, params={"pdf_id": pdf_id}) as ws:
                logger.info(f"Connected to chat for PDF ID: {pdf_id}")
                logger.info("Type 'exit' to quit the chat")

                # Main chat loop
                while True:
                    # Get user question
                    question = input("\nEnter Question: ")
                    if question.lower() == "exit":
                        break

                    # Send the question to the websocket
                    await ws.send_str(question)
                    logger.info("Waiting for response...")

                    # Receive and print the response
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            logger.info(f"Response: {msg.data}")
                            break  # Break after receiving the response
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("Connection closed")
                            return
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            return

                logger.info("Chat session ended")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    pdf_id = input("Enter PDF ID: ")
    asyncio.run(chat_with_pdf(pdf_id))
