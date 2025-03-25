import csv
import os
from pathlib import Path

import requests

# load_dotenv()

pswd = os.getenv("fire_api")
firecrawl_url = "https://api.firecrawl.dev/v1/scrape"

output_folder = Path("./temp_processing/output")
images_folder = output_folder / "extracted_images"
tables_folder = output_folder / "extracted_tables"
markdown_folder = output_folder / "markdown"


def firecrawl(url: str, job_name: str):
    payload = {
        "url": f"{url}",  # update the URL here
        "actions": [],
        "formats": ["markdown"],
    }
    headers = {"Authorization": pswd, "Content-Type": "application/json"}

    response = requests.request("POST", firecrawl_url, json=payload, headers=headers)

    data = response.json()

    # Ensure output directories exist .Change for cloud
    os.makedirs(markdown_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(tables_folder, exist_ok=True)

    # Save Markdown Content
    if data.get("success") and "markdown" in data["data"]:
        markdown_text = data["data"]["markdown"]
        md_file_path = markdown_folder / f"{job_name}.md"
        with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_text)
        print(f"Markdown saved to {md_file_path}")

    # Download and Save Images
    if data.get("success") and "markdown" in data["data"]:
        import re

        # Extract image URLs using regex
        image_urls = re.findall(r"!\[.*?\]\((https?://[^\s]+)\)", markdown_text)

        for image_count, img_url in enumerate(image_urls):
            img_path = images_folder / f"{job_name}-img{image_count}.png"

            # Download and save the image
            img_response = requests.get(img_url, stream=True)
            if img_response.status_code == 200:
                with open(img_path, "wb") as img_file:
                    for chunk in img_response.iter_content(1024):
                        img_file.write(chunk)
                print(f"Image saved: {img_path}")

    # Extract and Save CSV Data (if available)
    if "tables" in data["data"]:
        for i, table in enumerate(data["data"]["tables"]):
            csv_file_path = tables_folder / f"{job_name}-table{i}.csv"
            with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(table["headers"])
                writer.writerows(table["rows"])
            print(f"CSV saved: {csv_file_path}")


if __name__ == "__main__":
    firecrawl(
        "https://www.crummy.com/software/BeautifulSoup/bs4/doc/", "firecrawl_output"
    )
