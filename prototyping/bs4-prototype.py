import os
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup, ResultSet


class WebScraper:
    def __init__(self, url: str, job_name=None):
        self.url = url
        webpage = self.get_webpage()
        self.soup = BeautifulSoup(webpage, "lxml")
        suffix = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.job_name = f"{job_name}-{suffix}" if job_name else f"untitled-job-{suffix}"
        self.out_path = f"output/webscraper/{self.job_name}"
        os.makedirs(self.out_path, exist_ok=True)

    def extract_all(self):
        tables, images = self.soup.find_all("table"), self.soup.find_all("img")
        self.extract_modify_tables(tables)
        self.extract_modify_images(images)
        with open(self.out_path + "/index.html", "w") as file:
            file.write(str(self.soup))
        return self.out_path

    def get_webpage(self):
        response = requests.get(self.url)
        return response.text

    def extract_modify_images(self, images):
        for image_count, image in enumerate(images):
            image_url = image["src"]
            img_data = requests.get(self.url + "/" + image_url).content
            with open(
                f"{self.out_path}/images/image{image_count}.jpg", "wb"
            ) as handler:
                handler.write(img_data)
            new_tag = self.soup.new_tag("p")
            new_tag.string = f"[Image {image_count}](images/image{image_count}.jpg)"
            image.replace_with(new_tag)

    def extract_modify_tables(self, tables: ResultSet):
        for table_count, table in enumerate(tables):
            read_table = pd.read_html(str(table))
            df = read_table[0]
            df.to_csv(f"{self.out_path}/tables/table{table_count}.csv")
            new_tag = self.soup.new_tag("p")
            new_tag.string = f"[Table {table_count}](tables/table{table_count}.csv)"
            table.replace_with(new_tag)

    # def extract_table(self, count, child):
    #     table = pd.read_html(str(child))
    #     df = table[0]
    #     df.to_csv(f'{self.out_path}/table{count}.csv')

    # def process_element(self, count, child):
    #     # print(child.name)
    #     match child.name:
    #         case 'img':
    #             extract_img(count, child)
    #         case 'table':
    #             self.extract_table(count, child)
    #         case None:
    #             pass
    #         case _:
    #             # self.extract_text(count, child)
    #             print(child.name)
    #             pass


# def extract_links(count, child):
#     pass
#
#
# def extract_paragraph(count, child):
#     pass
#
#
# def extract_img(count, child):
#     pass


if __name__ == "__main__":
    modified_html = WebScraper(
        "https://www.crummy.com/software/BeautifulSoup/bs4/doc/", "bs4docs"
    ).extract_all()
