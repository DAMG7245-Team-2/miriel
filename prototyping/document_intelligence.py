import os

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# load_dotenv()

# Set up the client
endpoint = os.getenv("endpoint")
key = os.getenv("key")
credential = AzureKeyCredential(key)
client = DocumentAnalysisClient(endpoint, credential)

# Load the PDF file
with open(os.getenv("pdf_location"), "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", document=f)

# Get the results
result = poller.result()

# Create a Markdown file
with open("output.md", "w", encoding="utf-8") as md_file:
    md_file.write("# Extracted Text\n")
    for page in result.pages:
        for line in page.lines:
            md_file.write(f"{line.content}\n")

    md_file.write("\n# Extracted Tables\n")
    for table in result.tables:
        md_file.write(f"## Table {table.row_count}x{table.column_count}\n")
        for cell in table.cells:
            md_file.write(
                f"- Cell [{cell.row_index}, {cell.column_index}]: {cell.content}\n"
            )

    md_file.write("\n# Extracted Images\n")
    for page in result.pages:
        if page.selection_marks:
            for selection_mark in page.selection_marks:
                md_file.write(f"- Selection Mark: {selection_mark.state}\n")

    md_file.write("\n# Extracted Formulas\n")
    for page in result.pages:
        for formula in page.formulas:
            md_file.write(f"- Formula: {formula.content}\n")

print("Markdown file saved as 'output.md'.")
