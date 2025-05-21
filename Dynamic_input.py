import os
from typing import List
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(filename="document_router.log", level=logging.INFO)

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

# System prompt for document classification
system_prompt = """
You are a document classifier. Your task is to classify input paths into different agents based on these rules:

- If the input starts with 'http' or 'https', classify it as 'SharePoint Agent'.
- Else, if it's a local file path, classify based on its extension:
  - If it ends with '.pdf', '.docx', '.ppt', '.pptx', '.xlsx', classify as 'File Parsing Agent'.
  - If it ends with '.eml', classify as 'Email Agent'.
  - If it ends with '.vtt', '.txt', classify as 'Transcript Agent'.
  - If it ends with '.jpg', '.png', '.gif', classify as 'Image Processing Agent'.
  - For any other extension, classify as 'File Parsing Agent'.

Return a JSON object with 'agent' and 'path' fields, e.g., {'agent': 'File Parsing Agent', 'path': '/path/to/file.pdf'}
"""

# Task to classify documents using @task.agent without tools
@task.agent(model="gpt-4", result_type=Assignment, system_prompt=system_prompt)
def classify_document(input_path: str) -> Assignment:
    """
    Classify a document or URL to an agent based on the system prompt.
    Args:
        input_path: Path to a local file or a SharePoint URL.
    Returns:
        Assignment object with agent name and path/URL.
    """
    pass

# DAG definition
with DAG(
    dag_id="document_classifier",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
):
    # Example inputs (local files and SharePoint URLs)
    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/teams/tensaiGPT-PROD-HR-Docs/_api/web/GetFileByServerRelativeUrl('/teams/tensaiGPT-PROD-HR-Docs/Shared%20Documents/document1.pdf')",
        "/path/to/example.eml",
        "/path/to/transcript.vtt",
        "/path/to/image.jpg",
        "/path/to/unknown.xyz"
    ]

    # Classify each input using dynamic task mapping
    classifications = classify_document.expand(input_path=inputs)
