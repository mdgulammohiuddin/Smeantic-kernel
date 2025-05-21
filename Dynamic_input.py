import os
from typing import List
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

# Configure logging, inspired by reference code
logging.basicConfig(filename="document_router.log", level=logging.DEBUG)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('airflow_ai_sdk').setLevel(logging.DEBUG)

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
  - If it ends with '.pdf', '.pptx', '.xlsx', also classify as 'Image Processing Agent' (these files may contain images).
  - If it ends with '.eml' or '.msg', classify as 'Email Agent'.
  - If it ends with '.vtt' or '.txt', classify as 'Transcript Agent'.
  - If it ends with '.jpg', '.png', or '.gif', classify as 'Image Processing Agent'.
  - For any other extension, classify as 'File Parsing Agent'.

For inputs that match multiple criteria (e.g., '.pptx' for both 'File Parsing Agent' and 'Image Processing Agent'), return multiple JSON objects, one for each agent.

Return a list of JSON objects, each with 'agent' and 'path' fields, e.g., [{'agent': 'File Parsing Agent', 'path': '/path/to/file.pptx'}, {'agent': 'Image Processing Agent', 'path': '/path/to/file.pptx'}]
"""

@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['document_classification', 'airflow-ai-sdk', 'pydantic-ai'],
)
def document_classifier():
    """
    DAG to classify local files and SharePoint URLs into agents using Airflow AI SDK.
    """

    # Task to classify documents using @task.agent without tools
    @task.agent(model="gpt-4o", result_type=List[Assignment], system_prompt=system_prompt)
    def classify_document(input_path: str) -> List[Assignment]:
        """
        Classify a document or URL to one or more agents based on the system prompt.
        Args:
            input_path: Path to a local file or a SharePoint URL.
        Returns:
            List of Assignment objects with agent name and path/URL.
        """
        pass

    # Example inputs (local files and SharePoint URLs)
    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/teams/tensaiGPT-PROD-HR-Docs/_api/web/GetFileByServerRelativeUrl('/teams/tensaiGPT-PROD-HR-Docs/Shared%20Documents/document1.pdf')",
        "/path/to/example.eml",
        "/path/to/message.msg",
        "/path/to/transcript.vtt",
        "/path/to/image.jpg",
        "/path/to/unknown.xyz"
    ]

    # Classify each input using dynamic task mapping
    classifications = classify_document.expand(input_path=inputs)

# Instantiate the DAG
document_classifier_dag = document_classifier()
