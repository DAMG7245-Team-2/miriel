import os

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.generic.blank import Blank
from diagrams.onprem.client import Users, Client
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python

# Add Graphviz to PATH (if not already added)
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

# Create the diagram and save it as an image
diagram_filename = "architecture_diagram.png"
with Diagram(
    "Streamlit PDF/URL Parser Architecture",
    filename="architecture_diagram",
    show=False,
    direction="LR",
):
    # User
    user = Users("User")

    # Streamlit Frontend
    frontend = Client("Streamlit Frontend")

    # Backend Processing
    with Cluster("Backend Processing"):
        python_parser = Python("Python-Based Parser- Pdfplumber and dockling")
        llamaparser = Server("llamaparser")
        s3 = S3("S3 Bucket")

    # Output
    output = Blank("Markdown File")

    # Connections

    user >> Edge(label="PDF/URL Input") >> frontend
    frontend >> Edge(label="Fast API") >> llamaparser
    python_parser >> Edge(label="Extracted Data") >> output
    llamaparser >> Edge(label="Extracted Data") >> output
    llamaparser >> Edge(label="Store Images/Tables") >> s3
    python_parser >> Edge(label="Store Images/Tables") >> s3
    # output >> Edge(label="Markdown file") >> frontend
    # frontend >> Edge(label="Output file") >> user
