import os
from typing import List, Dict, Any
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent
from dotenv import load_dotenv
from airflow.utils.log.logging_mixin import LoggingMixin
from pdf2image import convert_from_path
from pptx import Presentation
from openpyxl import load_workbook

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required")
os.environ["OPENAI_API_KEY"] = api_key

# Logger
logger = LoggingMixin().log

# Model to structure the output
class ClassificationResult(BaseModel):
    agent: str = Field(description="Target agent name")
    path: str = Field(description="Input file or URL")

# Tool function
def detect_images_in_document(path: str) -> bool:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            images = convert_from_path(path)
            return len(images) > 0
        elif ext == ".pptx":
            prs = Presentation(path)
            return any(shape.shape_type == 13 for slide in prs.slides for shape in slide.shapes)
        elif ext == ".xlsx":
            wb = load_workbook(path)
            return any(sheet._images for sheet in wb)
        return False
    except Exception as e:
        logger.error(f"Error detecting images in {path}: {e}")
        return False

# System prompt for agent
SYSTEM_PROMPT = """
You are a document classifier. Based on the input path (file path or URL), classify which agents should process the document.

- For '.pdf', '.pptx', '.xlsx': always classify as 'File Parsing Agent'
- For those same files, if detect_images_in_document returns true or it's a SharePoint URL, also classify as 'Image Processing Agent'
- For '.eml', '.msg': classify as 'Email Agent'
- For '.vtt', '.txt': classify as 'Transcript Agent'
- For image extensions ('.jpg', '.png', '.gif'): classify as 'Image Processing Agent'
- For unknown or no extension: default to 'File Parsing Agent'
- If the path is a SharePoint URL (starts with 'http'), assume it may contain images

Always return a list of classification objects like:
[{"agent": "File Parsing Agent", "path": "<input>"}, {"agent": "Image Processing Agent", "path": "<input>"}]
"""

# AI agent for classification
document_classifier_agent = PydanticAIAgent(
    model="gpt-4o",
    system_prompt=SYSTEM_PROMPT,
    tools=[detect_images_in_document],
    output_model=List[ClassificationResult],
)

@dag(
    dag_id="document_classifier",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["document_classification"],
)
def document_classifier():

    @task.agent(agent=document_classifier_agent)
    def classify(input_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Running classifier for: {input_path}")
        results = document_classifier_agent.run_sync(input_path)
        return [res.model_dump() for res in results]

    @task
    def flatten(results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [item for sublist in results for item in sublist]

    @task
    def process(assignment: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Assignment: Agent={assignment['agent']}, Path={assignment['path']}")
        return {**assignment, "status": "processed"}

    @task
    def summarize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Final Summary:")
        for result in results:
            logger.info(result)
        return results

    # Input documents/URLs
    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/:b:/t/tensaiGPT-PROD-HR-Docs/example",
    ]

    classifications = classify.expand(input_path=inputs)
    flat = flatten(classifications)
    processed = process.expand(assignment=flat)
    summarize(processed)

doc_classifier_dag = document_classifier()
