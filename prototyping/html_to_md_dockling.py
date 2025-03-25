from docling.document_converter import DocumentConverter
from webscraper import WebScraper


def html_to_md(url: str, job_name: str):
    doc_converter = DocumentConverter()
    out = WebScraper(url, job_name).extract_all()
    conv_result = doc_converter.convert(out.get("html"))

    # Export to Markdown
    markdown_output = conv_result.document.export_to_markdown()

    markdown_path = f"{out.get('home')}/output.md"
    with open(markdown_path, "w") as f:
        f.write(markdown_output)
    return markdown_path
