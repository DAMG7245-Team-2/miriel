import uuid
import datetime
import requests
import logging

import streamlit as st
from streamlit import session_state as ss
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="Miriel", page_icon="💬", layout="wide")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

host = os.getenv("HOST")

if "chats" not in ss:
    ss.chats = {
        "1": {
            "id": "1",
            "name": "nvidia",
            "created_at": "2025-03-22 12:00:00",
            "has_pdf": True,
            "pdf_id": "1",
            "pdf_name": "nvidia.pdf",
            "summary_generated": True,
            "model": "gemini/gemini-2.0-flash",
        }
    }

if "messages" not in ss:
    ss.messages = {
        "1": [
            {
                "role": "system",
                "content": "Ask me any question about nvidia financial data",
            }
        ]
    }

if "current_chat_id" not in ss:
    ss.current_chat_id = None

if "new_chat_name" not in ss:
    ss.new_chat_name = ""

if "input_tokens" not in ss:
    ss.input_tokens = 0

if "output_tokens" not in ss:
    ss.output_tokens = 0

if "cost" not in ss:
    ss.cost = 0

if "year" not in ss:
    ss.year = None

if "quarter" not in ss:
    ss.quarter = None


def upload_pdf_to_backend(file, chat_id, ocr_method: str):
    try:
        logger.info(f"Uploading PDF to backend with OCR method: {ocr_method}")
        response = requests.post(
            f"{host}/upload_pdf",
            files={"file": (file.name, file, "application/pdf")},
            params={
                "parser": ocr_method,
                "chunking_strategy": chunking_strategy,
                "vector_store": vector_store,
            },
        )
        if response.status_code == 201:
            response_data = response.json()
            # Store the PDF ID and update chat status
            ss.chats[chat_id].update(
                {
                    "pdf_id": response_data["pdf_id"],
                    "has_pdf": True,
                    "pdf_name": file.name,
                }
            )
            logger.info(f"PDF uploaded to backend: {response_data['pdf_id']}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error uploading PDF to backend: {str(e)}")
        st.error("Error uploading PDF to backend")
        return False


def create_new_chat():
    if not ss.new_chat_name:
        return

    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ss.chats[chat_id] = {
        "id": chat_id,
        "name": ss.new_chat_name,
        "created_at": timestamp,
        "has_pdf": False,  # Track if PDF is uploaded
        "summary_generated": False,  # Track if summary has been generated
        "model": "gemini/gemini-2.0-flash",
    }
    logger.info(f"Created new chat: {ss.chats[chat_id]}")
    ss.current_chat_id = chat_id
    ss.messages[chat_id] = []
    ss.messages[chat_id].append(
        {
            "role": "system",
            "content": "Please upload a PDF document to begin the conversation.",
        }
    )

    # Clear the input field
    ss.new_chat_name = ""


def select_chat(chat_id):
    ss.current_chat_id = chat_id


def send_message():
    if not ss.user_input or not ss.current_chat_id:
        return

    if len(ss.user_input) < 10:
        ss.messages[ss.current_chat_id].extend(
            [
                {"role": "user", "content": ss.user_input},
                {
                    "role": "system",
                    "content": "Please enter a message longer than 10 characters.",
                },
            ]
        )
        return

    chat_id = ss.current_chat_id
    user_message = ss.user_input
    pdf_id = ss.chats[chat_id].get("pdf_id")
    logger.info(
        f"Creating new message for chat with id: {chat_id}, user message: {user_message}, pdf_id: {pdf_id}"
    )
    # if not pdf_id:
    #     logger.error("No PDF associated with this chat. Please upload a PDF first.")
    #     st.error("No PDF associated with this chat. Please upload a PDF first.")
    #     return

    # Add user message to chat
    ss.messages[chat_id].append({"role": "user", "content": user_message})

    try:
        response = {}
        if chat_id == "1":
            params_payload = (
                {
                    "year": ss["year"],
                    "quarter": ss["quarter"],
                }
                if hybrid_search
                else {}
            )
            print(params_payload)
            response = requests.post(
                f"{host}/ask_nvidia",
                json={
                    "pdf_id": "0000",
                    "question": user_message,
                    "max_tokens": 500,  # You can adjust this value
                },
                params=params_payload,
            )

        else:
            # Send question to backend with PDF ID
            response = requests.post(
                f"{host}/ask_question",
                params={
                    "pdf_id": pdf_id,
                    "question": user_message,
                    "max_tokens": 500,  # You can adjust this value
                    "model": ss.chats[chat_id]["model"],
                },
            )

        if response.status_code == 200:
            response_data = response.json()
            answer = response_data.get(
                "answer", "Sorry, I couldn't process your question."
            )

            # Display token usage and cost metrics
            usage_metrics = response_data.get("usage_metrics", {})
            if usage_metrics:
                ss.input_tokens += usage_metrics.get("input_tokens", 0)
                ss.output_tokens += usage_metrics.get("output_tokens", 0)
                ss.cost += usage_metrics.get("cost", 0)
            ss["messages"][chat_id].append({"role": "assistant", "content": answer})
            logger.info(f"Answer received from backend: {answer}")
        else:

            ss["messages"][chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your question.",
                }
            )
            logger.error(
                f"Error processing your question: {response.status_code}, {response.text}"
            )
    except Exception as e:
        logger.error(f"Error processing your question: {str(e)}")
        ss.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error processing your question: {str(e)}",
            }
        )


def insert_column_data(data_item):
    if not ss.current_chat_id:
        return

    chat_id = ss.current_chat_id

    ss.messages[chat_id].append(
        {"role": "user", "content": f"Tell me about: {data_item}"}
    )

    bot_response = (
        f"Here's information about {data_item}. This is a placeholder response."
    )
    ss.messages[chat_id].append({"role": "assistant", "content": bot_response})


def delete_chat(chat_id):
    logger.info(f"Deleting chat with id: {chat_id}")
    if chat_id in ss.chats:
        del ss.chats[chat_id]
    if chat_id in ss.messages:
        del ss.messages[chat_id]
    if ss.current_chat_id == chat_id:
        ss.current_chat_id = None
    st.rerun()


def clear_all_data():
    logger.info("Clearing all chat data")
    ss.chats = {}
    ss.messages = {}
    ss.current_chat_id = None
    st.rerun()


def generate_summary(chat_id):
    pdf_id = ss.chats[chat_id].get("pdf_id")

    if not pdf_id:
        logger.error("No PDF associated with this chat. Please upload a PDF first.")
        st.error("No PDF associated with this chat. Please upload a PDF first.")
        return

    try:
        response = requests.post(
            f"{host}/ask_question",
            params={
                "pdf_id": pdf_id,
                "question": "Summarize the document",
                "max_tokens": 500,  # You can adjust this value
                "model": ss.chats[chat_id]["model"],
            },
        )
        logger.info(f"Summary response: {response.json()}")
        if response.status_code == 200:
            response_data = response.json()
            summary = response_data.get(
                "answer", "Sorry, I couldn't process your question."
            )

            ss.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": f"Here's a summary of the document:\n\n{summary}",
                }
            )
            logger.info(f"Summary generated: {summary}")
            usage_metrics = response_data.get("usage_metrics", {})
            if usage_metrics:
                ss.input_tokens += usage_metrics.get("input_tokens", 0)
                ss.output_tokens += usage_metrics.get("output_tokens", 0)
                ss.cost += usage_metrics.get("cost", 0)
            # Mark summary as generated
            st.session_state.chats[chat_id]["summary_generated"] = True
        else:
            ss.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error generating the summary.",
                }
            )
            logger.error(
                f"Error generating summary: {response.status_code}, {response.text}"
            )
    except Exception as e:
        ss.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error generating summary: {str(e)}",
            }
        )
        logger.error(f"Error generating summary: {str(e)}")


left_col, middle_col, right_col = st.columns([20, 60, 20])

with left_col:
    st.header("Create New Chat")

    st.text_input(
        "Chat Name",
        key="new_chat_name",
        on_change=create_new_chat,
    )

    st.divider()
    st.subheader("Your Chats")

    if not ss.chats:
        st.info("No chats yet. Create a new chat to get started!")

    for chat_id, chat in ss.chats.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                f"{chat['name']}",
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                select_chat(chat_id)
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)

    if ss.chats:
        st.divider()
        if st.button("Clear All Chats", type="secondary"):
            clear_all_data()

with middle_col:
    st.header("Chat")

    chat_container = st.container(height=500, border=True)

    with chat_container:
        if ss.current_chat_id:
            chat_id = ss.current_chat_id
            chat = ss.chats[chat_id]
            user_input = ss["user_input"] = ""

            st.subheader(f"Chat: {chat['name']}")

            # Show PDF upload prompt if no PDF is uploaded yet
            if not chat.get("has_pdf", False):
                with st.status("Upload a PDF document", expanded=True) as upload_status:
                    ocr_method = st.selectbox(
                        "Select parsing method",
                        options=["mistral", "docling"],
                        index=0,
                    )
                    logger.info(f"Selected OCR method: {ocr_method}")
                    chunking_strategy = st.selectbox(
                        "Select chunking strategy",
                        options=["recursive", "kamradt", "fixed"],
                        index=0,
                    )
                    logger.info(f"Selected chunking strategy: {chunking_strategy}")
                    vector_store = st.selectbox(
                        "Select vector store",
                        options=["pinecone", "chroma", "naive"],
                        index=1,
                    )
                    logger.info(f"Selected vector store: {vector_store}")

                    uploaded_file = st.file_uploader(
                        "Upload a PDF document",
                        type=["pdf"],
                    )
                    if uploaded_file is not None:
                        if upload_pdf_to_backend(uploaded_file, chat_id, ocr_method):
                            ss.messages[chat_id].append(
                                {
                                    "role": "system",
                                    "content": f"PDF '{uploaded_file.name}' has been uploaded and processed. You can now ask questions about the document.",
                                }
                            )
                            upload_status.update(
                                label="PDF uploaded successfully!",
                                state="complete",
                                expanded=False,
                            )
                            ss.chats[chat_id]["has_pdf"] = True
                        else:
                            upload_status.update(
                                label="Failed to upload PDF",
                                state="error",
                            )
                            ss.chats[chat_id]["has_pdf"] = False
                        st.rerun()
            else:

                st.info(
                    f"📄 Current PDF: {chat.get('pdf_name')} (ID: {chat.get('pdf_id')})"
                )

                # Show messages
                for message in ss.messages.get(ss.current_chat_id, []):
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    elif message["role"] == "system":
                        st.chat_message("system").write(message["content"])
                    else:
                        st.chat_message("assistant").write(message["content"])

                if st.button(
                    "Summarize Document",
                    type="primary",
                    disabled=chat.get("summary_generated", False),
                ):
                    generate_summary(chat_id)
                    st.rerun()
                ss.chats[chat_id]["model"] = st.selectbox(
                    "Select model",
                    options=[
                        "gemini/gemini-2.0-flash",
                        "openai/gpt-4o-mini",
                        "xai/grok-2-latest",
                        "deepseek/deepseek-chat",
                        "anthropic/claude-3-5-sonnet-20240620",
                    ],
                    index=0,
                )
                if ss.current_chat_id == "1":
                    hybrid_search = st.checkbox("Hybrid Search")
                    if hybrid_search:
                        year = st.slider(
                            "Year",
                            min_value=2022,
                            max_value=2025,
                            value=2025,
                            step=1,
                            key="year",
                        )
                        quarter = st.selectbox(
                            "Quarter",
                            options=["Q1", "Q2", "Q3", "Q4"],
                            index=0,
                            key="quarter",
                        )
                st.text_input(
                    "Ask a question about the document",
                    key="user_input",
                    on_change=send_message,
                )
        else:
            st.markdown("### Create or select a chat to continue")
            st.markdown(
                "👈 Use the left panel to create a new chat or select an existing one"
            )

with right_col:
    st.header("Usage Metrics")
    with st.expander("📊 Summary Token Usage & Cost", expanded=True):
        st.metric("Input Tokens", ss.get("input_tokens", "N/A"))
        st.metric("Output Tokens", ss.get("output_tokens", "N/A"))
        st.metric("Cost ($)", f"${ss.get('cost', 0):.4f}")
