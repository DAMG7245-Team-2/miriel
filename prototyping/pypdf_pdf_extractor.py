import os

import camelot
from pypdf import PdfReader

# Input PDF file
pdf_file = "./example.pdf"

# Output files and folders
output_file = "pypdf_extracted_output.md"
image_folder = "pypdf_extracted_images"
table_folder = "pypdf_extracted_tables"

# PDF reader object
reader = PdfReader(pdf_file)


# Extract text and images together, rendering images in place
def extract_text_and_images_to_markdown(reader, output_file, image_folder):
    os.makedirs(image_folder, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for page_number, page in enumerate(reader.pages):
            f.write(f"# Page {page_number + 1}\n\n")

            # Extract text
            text = page.extract_text()
            if text:
                f.write(text + "\n\n")
            else:
                f.write("No text found on this page.\n\n")

            # Extract images
            for image_index, image_object in enumerate(page.images):
                image_name = f"page_{page_number + 1}_image_{image_index + 1}_{image_object.name}"
                image_path = os.path.join(image_folder, image_name)

                with open(image_path, "wb") as image_file:
                    image_file.write(image_object.data)

                f.write(f"![Image](./{image_folder}/{image_name})\n\n")  # Link to image
        print(
            f'Text and image extracted and saved to "{output_file}". Images saved to "{image_folder}".'
        )


# Extract tables from the PDF and add links to Markdown
def extract_tables_from_pdf(pdf_file, output_file, table_folder):
    os.makedirs(table_folder, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        try:
            # Extract tables using Camelot
            tables = camelot.read_pdf(
                pdf_file, pages="all", flavor="lattice"
            )  # Try flavor="stream" later

            if len(tables) == 0:
                f.write("No tables found in the PDF.\n\n")
                print("No tables found.")
                return

            # Saving each table as a CSV file and adding a link in the Markdown file
            for i, table in enumerate(tables):
                table_name = f"table_{i + 1}.csv"
                table_path = os.path.join(table_folder, table_name)
                table.to_csv(table_path)

                # Link
                f.write(f"[Table {i + 1}](./{table_folder}/{table_name})\n\n")

            print(f"Table extraction completed. Tables saved to '{table_folder}'.")
        except Exception as e:
            print(f"Error extracting tables: {e}")


# Main function to perform complete extraction
def main():
    # Extract text and images together to Markdown file
    extract_text_and_images_to_markdown(reader, output_file, image_folder)

    # Extract tables and add links to Markdown file
    extract_tables_from_pdf(pdf_file, output_file, table_folder)


if __name__ == "__main__":
    main()
