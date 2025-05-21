import os
from typing import List, Dict, Any
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent
from dotenv import load_dotenv
import logging
import sys
from airflow.utils.log.logging_mixin import LoggingMixin
from pdf2image import convert_from_path
from pptx import Presentation
from openpyxl import load_workbook

# Configure logging for Airflow UI and file
logger = logging.getLogger('airflow.task')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('/app/fdi/airflow/logs/document_router.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('airflow').setLevel(logging.DEBUG)

# Load .env file and set OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Pydantic model for assignment output
class Assignment(BaseModel):
    agent: str = Field(description="Name of the assigned agent")
    path: str = Field(description="Path or URL of the document")

# Tool function to detect images in documents
def detect_images_in_document(file_path: str) -> bool:
    """
    Detects if a document (.pdf, .pptx, .xlsx) contains images.
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
        logger.error(f"Error detecting images in {file_path}: {e}")
        return False

# Define the system prompt for document classification
SYSTEM_PROMPT = """
You are a document classifier. Analyze the input path (a string representing a local file path or SharePoint URL) and determine appropriate agents based on these rules:

- Determine the file extension:
  - For local paths, use the extension (e.g., '.pptx' from '/path/to/file.pptx').
  - For URLs starting with 'http' or 'https', extract the extension from the path (e.g., '.pdf' from '/Shared%20Documents/document1.pdf' or ':b:/t/...'). Assume '.pdf' if unclear.
- Classify based on the extension:
  - '.pdf', '.docx', '.ppt', '.pptx', '.xlsx' → File Parsing Agent.
  - '.pdf', '.pptx', '.xlsx' → Image Processing Agent if 'detect_images_in_document' returns True (local files only; assume True for URLs).
  - '.eml', '.msg' → Email Agent.
  - '.vtt', '.txt' → Transcript Agent.
  - '.jpg', '.png', '.gif' → Image Processing Agent.
  - Other/no extension → File Parsing Agent.
  - URLs → SharePoint Agent.
- For local '.pdf', '.pptx', '.xlsx', use 'detect_images_in_document' to check for images.

For inputs matching multiple criteria (e.g., '.pptx' with images), return multiple assignments.

Input: A single string (file path or URL).
Output: A list of JSON objects, each with 'agent' and 'path' fields, e.g., [{'agent': 'File Parsing Agent', 'path': '/path/to/file.pptx'}, {'agent': 'Image Processing Agent', 'path': '/path/to/file.pptx'}].
"""

# Create a Pydantic AI agent instance for document classification
document_classifier_agent = PydanticAIAgent(
    model="gpt-4",
    system_prompt=SYSTEM_PROMPT,
    tools=[detect_images_in_document],
    result_type=List[Assignment]
)

@dag(
    dag_id="document_classifier",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["document_classification", "airflow-ai-sdk"],
)
def document_classifier():
    """
    DAG to classify local files and SharePoint URLs into agent assignments.
    """

    @task.agent(agent=document_classifier_agent)
    def classify_document(input_path: str) -> List[Dict[str, Any]]:
        """
        Classify a document or URL into one or more agent assignments.
        """
        logger.debug(f"Classifying input: {input_path}")
        result = document_classifier_agent.run_sync(input_path)
        logger.debug(f"Agent output: {[r.model_dump() for r in result]}")
        return [assignment.model_dump() for assignment in result]

    @task
    def flatten_assignments(list_of_assignments: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Flatten a list of lists of assignments into a single list.
        """
        flat_list = []
        for sublist in list_of_assignments:
            flat_list.extend(sublist)
        return flat_list

    @task
    def process_assignment(assignment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single assignment.
        """
        logger.info(f"Processing assignment - Agent: {assignment['agent']}, Path: {assignment['path']}")
        return {
            "agent": assignment['agent'],
            "path": assignment['path'],
            "status": "processed"
        }

    @task
    def aggregate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate and log all processing results.
        """
        logger.info("------ Processing Results ------")
        for result in results:
            logger.info(f"Agent: {result['agent']}, Path: {result['path']}, Status: {result['status']}")
        logger.info("------ End of Results ------")
        return results

    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/:b:/t/tensaiGPT-PROD-HR-Docs/ET0W0clrClhBrA7ZLzCoOmEBHq0vg-rFuGuEwb40Weq8zQ?e=6BkU3C",
    ]
    assignments_mapped = classify_document.expand(input_path=inputs)
    flattened = flatten_assignments(assignments_mapped)
    processed = process_assignment.expand(assignment=flattened)
    aggregate_results(processed)

doc_classifier_dag = document_classifier()
