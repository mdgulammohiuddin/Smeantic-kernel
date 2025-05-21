import os
from typing import List, Dict, Any
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent
from dotenv import load_dotenv
import logging
from pdf2image import convert_from_path
from pptx import Presentation
from openpyxl import load_workbook

# Configure logging
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

# Define the system prompt for document classification
SYSTEM_PROMPT = """
You are a document classifier. Analyze the input path and determine appropriate agents based on these rules:

1. File types and their agents:
   - PDF, DOCX, PPT, PPTX, XLSX → File Parsing Agent
   - PDF, PPTX, XLSX with images → Also Image Processing Agent
   - EML, MSG → Email Agent
   - VTT, TXT → Transcript Agent
   - JPG, PNG, GIF → Image Processing Agent
   - SharePoint URLs → Add SharePoint Agent

2. For local files (.pdf, .pptx, .xlsx):
   - Use detect_images_in_document tool to check for images
   - If images found, include Image Processing Agent

3. For SharePoint URLs:
   - Assume PDF format if extension unclear
   - Assume images present unless specified otherwise

Return a list of assignments with agent and path for each applicable agent.
"""

# Create a Pydantic AI agent instance for document classification
document_classifier_agent = PydanticAIAgent(
    model="gpt-4",
    system_prompt=SYSTEM_PROMPT,
    result_type=List[Assignment]
)

# Use task.agent to decorate the classify_document task
@task.agent(agent=document_classifier_agent)
def classify_document(input_path: str) -> List[Dict[str, Any]]:
    """
    Classify a document or URL into one or more agent assignments.
    This function uses the Pydantic AI agent to return a list of assignments.
    """
    # Check for images if it's a local file with supported extension
    has_images = False
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.pdf', '.pptx', '.xlsx']:
            has_images = detect_images_in_document(input_path)
    
    # Build assignments based on static rules (as fallback/default logic)
    assignments = []
    ext = os.path.splitext(input_path)[1].lower()
    # File Parsing Agent
    if ext in ['.pdf', '.docx', '.ppt', '.pptx', '.xlsx']:
        assignments.append(Assignment(agent="File Parsing Agent", path=input_path))
    # Image Processing Agent when images are detected or typical image extension
    if has_images or ext in ['.jpg', '.png', '.gif']:
        assignments.append(Assignment(agent="Image Processing Agent", path=input_path))
    # Email Agent
    if ext in ['.eml', '.msg']:
        assignments.append(Assignment(agent="Email Agent", path=input_path))
    # Transcript Agent
    if ext in ['.vtt', '.txt']:
        assignments.append(Assignment(agent="Transcript Agent", path=input_path))
    # SharePoint Agent for URLs
    if input_path.startswith(('http://', 'https://')):
        assignments.append(Assignment(agent="SharePoint Agent", path=input_path))
    
    # The Pydantic AI agent can override or augment these assignments.
    # Call the agent synchronously.
    return document_classifier_agent.run_sync(assignments)
 
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
    logging.info(f"Processing assignment - Agent: {assignment['agent']}, Path: {assignment['path']}")
    # Here you could add more processing logic based on assignment['agent']
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
    logging.info("------ Processing Results ------")
    for result in results:
        logging.info(f"Agent: {result['agent']}, Path: {result['path']}, Status: {result['status']}")
    logging.info("------ End of Results ------")
    return results

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
    inputs = [
        "/app/fdi/Documents/FRD.pptx",
        "https://hexawareonline.sharepoint.com/:b:/t/tensaiGPT-PROD-HR-Docs/ET0W0clrClhBrA7ZLzCoOmEBHq0vg-rFuGuEwb40Weq8zQ?e=6BkU3C",
    ]
    # Map classification over each input
    assignments_mapped = classify_document.expand(input_path=inputs)
    # assignments_mapped is a list (one element per input), each element is a list of assignments.
    # Flatten the mapped assignments.
    flattened = flatten_assignments(assignments_mapped)
    # Dynamically map task to process each assignment in the flattened list.
    processed = process_assignment.expand(assignment=flattened)
    aggregate_results(processed)

doc_classifier_dag = document_classifier()
