import os
from typing import List
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent
from dotenv import load_dotenv
import logging
from pdf2image import convert_from_path
from pptx import Presentation
from openpyxl import load_workbook

# Configure logging, inspired by Reference Code 1
logging.basicConfig(filename="document_router.log", level=logging.DEBUG)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('airflow').setLevel(logging.DEBUG)

# Load .env file and set OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    logging.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Pydantic model for assignment output
class Assignment(BaseModel):
    agent: str = Field(description="Name of the assigned agent")
    path: str = Field(description="Path or URL of the document")

# Tool function to detect images in documents
def detect_images_in_document(file_path: str) -> bool:
    """
    Detects if a document (.pdf, .pptx, .xlsx) contains images.
    Args:
        file_path: Path to the document (local file only).
    Returns:
        bool: True if images are found, False otherwise.
    """
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == '.pdf':
            images = convert_from_path(file_path)
            return len(images) > 0
        elif extension == '.pptx':
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                        return True
            return False
        elif extension == '.xlsx':
            wb = load_workbook(file_path)
            for sheet in wb:
                if sheet._images:
                    return True
            return False
        return False
    except Exception as e:
        logging.error(f"Error detecting images in {file_path}: {e}")
        return False

# System prompt for document classification
system_prompt = """
You are a document classifier. Your task is to classify input paths into different agents based on these rules:

- For each input (local file path or URL), determine the file extension:
  - For local file paths, use the extension directly (e.g., '.pptx' from '/path/to/file.pptx').
  - For URLs starting with 'http' or 'https', extract the file extension from the path component (e.g., '.pdf' from '/Shared%20Documents/document1.pdf' or from a SharePoint URL like ':b:/t/...'). If the extension is unclear, assume '.pdf' for SharePoint URLs.
- Classify based on the extension:
  - If it ends with '.pdf', '.docx', '.ppt', '.pptx', '.xlsx', classify as 'File Parsing Agent'.
  - If it ends with '.pdf', '.pptx', '.xlsx' and the 'detect_images_in_document' tool returns True for the file, also classify as 'Image Processing Agent'.
  - If it ends with '.eml' or '.msg', classify as 'Email Agent'.
  - If it ends with '.vtt' or '.txt', classify as 'Transcript Agent'.
  - If it ends with '.jpg', '.png', or '.gif', classify as 'Image Processing Agent'.
  - For any other extension or no extension, classify as 'File Parsing Agent'.
  - For URLs, also classify as 'SharePoint Agent' to indicate the source.
- For local files with '.pdf', '.pptx', or '.xlsx', use the 'detect_images_in_document' tool to check for images. For URLs with these extensions, assume images are present unless otherwise specified.

For inputs that match multiple criteria (e.g., '.pptx' with images), return multiple JSON objects, one for each agent.

Return a list of JSON objects, each with 'agent' and 'path' fields, e.g., [{'agent': 'File Parsing Agent', 'path': '/path/to/file.pptx'}, {'agent': 'Image Processing Agent', 'path': '/path/to/file.pptx'}]
"""

# Define the Pydantic AI agent for classification
classifier_pydantic_agent = PydanticAIAgent(
    model="gpt-4o",
    system_prompt=system_prompt,
    tools=[detect_images_in_document],
    result_type=List[Assignment]
)

@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['document_classification', 'pydantic-ai', 'airflow-ai-sdk'],
)
def document_classifier():
    """
    DAG to classify local files and SharePoint URLs into agents using Pydantic AI.
    """

    @task.agent(agent=classifier_pydantic_agent)
    def classify_document(input_path: str) -> List[dict]:
        """
        Classify a document or URL to one or more agents based on the system prompt.
        Args:
            input_path: Path to a local file or a SharePoint URL.
        Returns:
            List of dicts with agent name and path/URL.
        """
        result = classifier_pydantic_agent.run(input_path)
        return [assignment.model_dump() for assignment in result]

    @task
    def show_classifications(classifications: List[List[dict]]):
        """
        Display and log classification results.
        Args:
            classifications: List of lists of dicts.
        Returns:
            Flattened list of dicts.
        """
        logging.info("------ Classification Results ------")
        flattened = [assignment for sublist in classifications for assignment in sublist]
        for assignment in flattened:
            logging.info(f"Agent: {assignment['agent']}, Path: {assignment['path']}")
        logging.info("------ End of Classification Results ------")
        return flattened

    # Example inputs (local files and SharePoint URLs)
    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/:b:/t/tensaiGPT-PROD-HR-Docs/ET0W0clrClhBrA7ZLzCoOmEBHq0vg-rFuGuEwb40Weq8zQ?e=6BkU3C",
        "/path/to/example.eml",
        "/path/to/message.msg",
        "/path/to/transcript.vtt",
        "/path/to/image.jpg",
        "/path/to/unknown.xyz"
    ]

    # Classify each input and show results
    classifications = classify_document.expand(input_path=inputs)
    show_classifications(classifications)

# Instantiate the DAG
document_classifier_dag = document_classifier()
