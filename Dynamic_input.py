import os
from typing import Literal, List
from airflow.decorators import dag, task
from datetime import datetime
from pydantic import BaseModel, Field
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
from office365.runtime.client_request_exception import ClientRequestException
from dotenv import load_dotenv
import logging
import io

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

# Pydantic models
class DocumentMetadata(BaseModel):
    path: str = Field(description="Path or URL of the document")
    content_type: str = Field(description="MIME type or file extension")
    has_images: bool = Field(description="Whether the document contains images")
    is_sharepoint: bool = Field(description="Whether the document is from SharePoint")

class AgentAssignment(BaseModel):
    document: DocumentMetadata
    agent_type: Literal["file_parsing", "email", "transcript", "image_processing"]
    priority: int = Field(description="Processing priority (1-5)", ge=1, le=5)

@task
def validate_and_extract_metadata(source: str, is_sharepoint: bool = False) -> DocumentMetadata:
    """Validate source and extract document metadata"""
    if is_sharepoint:
        try:
            client_id = os.getenv("CLIENT_ID")
            client_secret = os.getenv("CLIENT_SECRET")
            site_url = "https://hexawareonline.sharepoint.com/teams/tensaiGPT-PROD-HR-Docs"
            ctx = ClientContext(site_url).with_credentials(
                ClientCredential(client_id, client_secret)
            )
            
            file = ctx.web.get_file_by_server_relative_url(
                source.split("GetFileByServerRelativeUrl")[1][1:-1]
            ).execute_query()
            
            return DocumentMetadata(
                path=source,
                content_type=file.properties["Name"].split(".")[-1].lower(),
                has_images=False,  # Will be updated by content analysis
                is_sharepoint=True
            )
        except Exception as e:
            logging.error(f"SharePoint validation failed: {str(e)}")
            raise
    
    if not os.path.exists(source):
        raise ValueError(f"Local file not found: {source}")
    
    return DocumentMetadata(
        path=source,
        content_type=source.split(".")[-1].lower(),
        has_images=False,  # Will be updated by content analysis
        is_sharepoint=False
    )

@task.llm(
    model="gpt-4",
    task_id="determine_agents",
    result_type=List[AgentAssignment],
    system_prompt="""
    You are a document routing expert that analyzes document metadata and assigns appropriate processing agents.
    Consider the following agents:
    - File Parsing Agent: Handles PDF, PPT, PPTX, DOCX for text extraction
    - Email Agent: Processes email (EML) files
    - Transcript Agent: Processes text-heavy documents and transcripts
    - Image Processing Agent: Handles documents with embedded images

    Analyze the document metadata and return the appropriate agent assignments.
    """
)
def determine_agents(metadata: DocumentMetadata) -> List[AgentAssignment]:
    """Use LLM to determine which agents should process the document"""
    # The LLM will analyze the metadata and return appropriate agent assignments
    # The actual implementation is handled by the task.llm decorator
    pass

@task(task_id="file_parsing_task")
def process_with_file_parsing_agent(assignment: AgentAssignment) -> dict:
    """Process document with File Parsing Agent"""
    logging.info(f"Processing {assignment.document.path} with File Parsing Agent")
    # Add file parsing logic here
    return {"status": "success", "agent": "file_parsing"}

@task(task_id="email_task")
def process_with_email_agent(assignment: AgentAssignment) -> dict:
    """Process document with Email Agent"""
    logging.info(f"Processing {assignment.document.path} with Email Agent")
    # Add email processing logic here
    return {"status": "success", "agent": "email"}

@task(task_id="transcript_task")
def process_with_transcript_agent(assignment: AgentAssignment) -> dict:
    """Process document with Transcript Agent"""
    logging.info(f"Processing {assignment.document.path} with Transcript Agent")
    # Add transcript processing logic here
    return {"status": "success", "agent": "transcript"}

@task(task_id="image_processing_task")
def process_with_image_processing_agent(assignment: AgentAssignment) -> dict:
    """Process document with Image Processing Agent"""
    logging.info(f"Processing {assignment.document.path} with Image Processing Agent")
    # Add image processing logic here
    return {"status": "success", "agent": "image_processing"}

@dag(
    dag_id="intelligent_document_router",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)
def document_router_dag():
    # Define document sources
    documents = ["/app/fdi/Documents/FRD.pptx"]
    sharepoint_urls = [
        "https://hexawareonline.sharepoint.com/teams/tensaiGPT-PROD-HR-Docs/Shared%20Documents/Forms/AllItems.aspx"
    ]
    
    # Process local documents
    local_metadata = [validate_and_extract_metadata(doc) for doc in documents]
    
    # Process SharePoint documents
    sharepoint_metadata = [
        validate_and_extract_metadata(url, is_sharepoint=True) 
        for url in sharepoint_urls
    ]
    
    # Combine all document metadata
    all_metadata = local_metadata + sharepoint_metadata
    
    # Determine agent assignments using LLM
    assignments = determine_agents.expand(metadata=all_metadata)
    
    # Route to appropriate agents
    for assignment in assignments:
        if assignment.agent_type == "file_parsing":
            process_with_file_parsing_agent(assignment)
        elif assignment.agent_type == "email":
            process_with_email_agent(assignment)
        elif assignment.agent_type == "transcript":
            process_with_transcript_agent(assignment)
        elif assignment.agent_type == "image_processing":
            process_with_image_processing_agent(assignment)

# Create DAG instance
dag_instance = document_router_dag()
