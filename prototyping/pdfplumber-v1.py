import io

import fitz
import matplotlib.pyplot as plt
import pdfplumber
from PIL import Image


def open_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            return io.BytesIO(f.read())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def extract_text_and_tables(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        all_text = ""
        tables = []
        for page in pdf.pages:
            all_text += page.extract_text()
            tables_on_page = page.extract_tables()
            if tables_on_page:
                tables.extend(tables_on_page)
    return all_text, tables


def extract_images(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    image_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list.extend(page.get_images(full=True))
    images = []
    for img in image_list:
        xref = img[0]
        base_image = doc.extract_image(xref)
        img_bytes = base_image["image"]
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)
    return images


def extract_charts(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    charts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))
            charts.append(image)
    return charts


def visualize_charts(charts):
    for idx, chart in enumerate(charts):
        plt.figure(idx)
        plt.imshow(chart)
        plt.axis("off")
        plt.title(f"Chart {idx + 1}")
        plt.show()


def process_pdf(file_path):
    pdf_file = open_pdf(file_path)
    if not pdf_file:
        return
    text, tables = extract_text_and_tables(pdf_file)
    print(text)
    print("Tables Extracted:", len(tables))

    images = extract_images(pdf_file)
    print(f"Found {len(images)} images in the PDF.")

    charts = extract_charts(pdf_file)
    print(f"Found {len(charts)} potential charts in the PDF.")

    # visualize_charts(charts)


pdf_file_path = "/path/example.pdf"
process_pdf(pdf_file_path)

"""
from pypdf import PdfReader

reader = PdfReader("example.pdf")

text = []

for page in reader.pages:
    print(page.extract_text())
    for count, image_file_object in enumerate(page.images):
        with open('extracted/'+str(count) + image_file_object.name, "wb") as fp:
            fp.write(image_file_object.data)
"""
