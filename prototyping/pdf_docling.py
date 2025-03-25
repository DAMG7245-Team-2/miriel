import json

from docling.document_converter import DocumentConverter


def convert_pdf_with_docling(source_pdf_path, output_file_path):
    # Intializing the DocumentConverter
    converter = DocumentConverter()

    # Convert the PDF
    result = converter.convert(source)

    # Print results to the console
    plain_text = result.document.export_to_text()

    # Saving results to a text file
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result.document.export_to_dict(), f, indent=4)

    return plain_text


if __name__ == "__main__":
    source = "./example.pdf"
    output = "docling_conversion_results.json"
    text_result = convert_pdf_with_docling(source, output)
    print(text_result)
