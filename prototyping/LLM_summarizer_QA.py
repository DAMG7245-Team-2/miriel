import litellm
import os
import PyPDF2

# load_dotenv()

api_keys = {
    "gpt-4o": os.getenv("OPENAI_API_KEY"),
    "gemini-flash": os.getenv("GOOGLE_API_KEY"),
    "deepseek-chat": os.getenv("DEEPSEEK_API_KEY"),
    "claude-3-haiku": os.getenv("ANTHROPIC_API_KEY"),
    "grok": os.getenv("X_API_KEY"),
}

# Model mappings to LiteLLM providers-Don't change the names
model_mappings = {
    "gpt-4o": "openai/gpt-4o",
    "gemini-flash": "gemini/gemini-2.0-flash",
    "deepseek-chat": "deepseek/deepseek-chat",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "grok": "xai/grok-1",
}


# Extracts text from a given PDF file
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip() if text else "Error: No text extracted from PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# Summarizes the extracted text using the selected LLM
def summarize_text(text, markdown_file, model_name):
    if model_name not in model_mappings:
        return {
            "error": "Invalid model selection. Choose from: "
            + ", ".join(model_mappings.keys())
        }

    model = model_mappings[model_name]
    api_key = api_keys[model_name]

    if not api_key:
        return {"error": f"API key for {model_name} is missing."}

    try:
        with open(markdown_file, "r", encoding="utf-8") as markdown_file:
            markdown_content = markdown_file.read()
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following text in 5 to 6 lines:",
                },
                {"role": "user", "content": markdown_content},
            ],
            api_key=api_key,
            stream=False,
        )

        return {
            "summary": response["choices"][0]["message"]["content"],
            "input_tokens": response["usage"]["prompt_tokens"],
            "output_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
            "model": model_name,
        }

    except Exception as e:
        return {"error": str(e)}


# Answers a question based on the extracted PDF text using the selected LLM
def answer_question_from_pdf(text, markdown_file, question, model_name):
    if model_name not in model_mappings:
        return {
            "error": "Invalid model selection. Choose from: "
            + ", ".join(model_mappings.keys())
        }

    model = model_mappings[model_name]
    api_key = api_keys[model_name]

    if not api_key:
        return {"error": f"API key for {model_name} is missing."}

    try:
        with open(markdown_file, "r", encoding="utf-8") as markdown_file:
            markdown_content = markdown_file.read()
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the following question based on the given text:",
                },
                {
                    "role": "user",
                    "content": f"Markdown Document: {markdown_content}\n\nQuestion: {question}",
                },
            ],
            api_key=api_key,
            stream=False,
        )

        return {
            "question": question,
            "answer": response["choices"][0]["message"]["content"],
            "input_tokens": response["usage"]["prompt_tokens"],
            "output_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        }

    except Exception as e:
        return {"error": str(e)}


# Usage
pdf_path = "Resume_Continental.pdf"  # Replace with the actual PDF file path
selected_model = "gemini-flash"  # Change to "gpt-4o" "gemini-flash", "deepseek-chat", "claude-3-haiku", or "grok"
markdown_file = "Output.md"
pdf_text = extract_text_from_pdf(pdf_path)
if pdf_text.startswith("Error"):
    print(pdf_text)
else:
    # Sumamrize
    result = summarize_text(pdf_text, markdown_file, selected_model)
    print("\nSummary\n", result.get("summary"))
    print("\nTotal No Of Tokens\n", result.get("total_tokens"))
    print("\nNo Of Input Tokens\n", result.get("input_tokens"))
    print("\nNo Of Output Tokens\n", result.get("output_tokens"))

    # Ask a question based on the PDF content
    user_question = "Name the authors of the paper?"  # Modify this question as needed
    answer_result = answer_question_from_pdf(
        pdf_text, markdown_file, user_question, selected_model
    )
    print("\nQuestion:", answer_result.get("question"))
    print("\nAnswer:", answer_result.get("answer"))
    print("\nTotal No Of Tokens:", answer_result.get("total_tokens"))
    print("\nNo Of Input Tokens\n", answer_result.get("input_tokens"))
    print("\nNo Of Output Tokens\n", answer_result.get("output_tokens"))
