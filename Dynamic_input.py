from airflow import DAG
from airflow.decorators import task
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, List
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
import os
import PyPDF2
from docx import Document
from pptx import Presentation
import eml_parser
from pdf2image import convert_from_path
import io
import logging

# Configure logging
logging.basicConfig(filename="document_router.log", level=logging.INFO)

# Pydantic model for document assignment
class DocumentAssignment(BaseModel):
    document_path: str = Field(description="Path or URL of the document")
    assigned_agent: Literal["File Parsing Agent", "Email Agent", "Transcript Agent", "Image Processing Agent"] = Field(description="Agent to process the document")

@task
def route_documents() -> List[DocumentAssignment]:
    # Hardcoded files and URLs
    DOCUMENTS = [
        "/path/to/document1.pdf",
        "/path/to/document2.docx",
        "/path/to/document3.pptx",
        "/path/to/document4.txt",
        "/path/to/email.eml",
        "/path/to/spreadsheet.xlsx",
        "/path/to/subtitle.vtt"
    ]
    SHAREPOINT_URLS = [
        "https://your-tenant.sharepoint.com/sites/your-site/_api/web/GetFileByServerRelativeUrl('/sites/your-site/Shared%20Documents/document1.pdf')",
        "https://your-tenant.sharepoint.com/sites/your-site/_api/web/GetFileByServerRelativeUrl('/sites/your-site/Shared%20Documents/document2.docx')"
    ]

    # SharePoint credentials (use Airflow Connections in production)
    site_url = "https://your-tenant.sharepoint.com/sites/your-site"
    username = "your-username"
    password = "your-password"

    def validate_source(source: str) -> bool:
        """Validate if the source (file or URL) is accessible."""
        if source.startswith("http"):
            try:
                ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
                file = ctx.web.get_file_by_server_relative_url(source.split("GetFileByServerRelativeUrl")[1][1:-1]).execute_query()
                return True
            except Exception as e:
                logging.error(f"Failed to validate SharePoint URL {source}: {str(e)}")
                return False
        return os.path.exists(source)

    def download_sharepoint_file(url: str) -> io.BytesIO:
        """Download SharePoint file to a BytesIO object."""
        ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
        file = ctx.web.get_file_by_server_relative_url(url.split("GetFileByServerRelativeUrl")[1][1:-1]).execute_query()
        content = io.BytesIO()
        file.download(content).execute_query()
        content.seek(0)
        return content

    def has_images_in_pdf(file_path: str) -> bool:
        """Check if a PDF contains images."""
        try:
            images = convert_from_path(file_path, first_page=1, last_page=1)  # Check first page for efficiency
            return len(images) > 0
        except Exception as e:
            logging.warning(f"Error checking images in PDF {file_path}: {str(e)}")
            return False

    def has_images_in_docx(file_path: str) -> bool:
        """Check if a DOCX contains images."""
        try:
            doc = Document(file_path)
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    return True
            return False
        except Exception as e:
            logging.warning(f"Error checking images in DOCX {file_path}: {str(e)}")
            return False

    def has_images_in_pptx(file_path: str) -> bool:
        """Check if a PPTX contains images."""
        try:
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                        return True
            return False
        except Exception as e:
            logging.warning(f"Error checking images in PPTX {file_path}: {str(e)}")
            return False

    def has_images_in_eml(file_path: str) -> bool:
        """Check if an EML contains image attachments."""
        try:
            with open(file_path, "r") as f:
                eml = eml_parser.EmlParser().decode_email(f.read())
                for attachment in eml.get("attachment", []):
                    if attachment["content_type"].startswith("image/"):
                        return True
                return False
        except Exception as e:
            logging.warning(f"Error checking images in EML {file_path}: {str(e)}")
            return False

    def assign_agent(source: str, is_sharepoint: bool = False) -> List[DocumentAssignment]:
        """Assign agents based on file extension and image content."""
        assignments = []
        extension = source.lower().split(".")[-1]

        # Handle SharePoint files by downloading to a temporary file
        temp_file = None
        if is_sharepoint:
            content = download_sharepoint_file(source)
            temp_file = f"/tmp/{source.split('/')[-1]}"
            with open(temp_file, "wb") as f:
                f.write(content.read())
            source = temp_file

        # Base assignments by extension
        if extension in ["pdf", "ppt", "pptx"]:
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="File Parsing Agent"
            ))
        elif extension == "eml":
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Email Agent"
            ))
        elif extension == "vtt":
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Transcript Agent"
            ))
        elif extension == "docx":
            assignments.extend([
                DocumentAssignment(
                    document_path=source,
                    assigned_agent="File Parsing Agent"
                ),
                DocumentAssignment(
                    document_path=source,
                    assigned_agent="Transcript Agent"
                )
            ])

        # Check for images and assign to Image Processing Agent
        if extension == "pdf" and has_images_in_pdf(source):
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Image Processing Agent"
            ))
        elif extension == "docx" and has_images_in_docx(source):
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Image Processing Agent"
            ))
        elif extension in ["ppt", "pptx"] and has_images_in_pptx(source):
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Image Processing Agent"
            ))
        elif extension == "eml" and has_images_in_eml(source):
            assignments.append(DocumentAssignment(
                document_path=source,
                assigned_agent="Image Processing Agent"
            ))

        # Clean up temporary file for SharePoint
        if is_sharepoint and temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

        return assignments

    # Process all sources
    results = []
    for source in DOCUMENTS:
        if validate_source(source):
            assignments = assign_agent(source)
            for assignment in assignments:
                logging.info(f"Assigned {assignment.document_path} to {assignment.assigned_agent}")
            results.extend(assignments)
        else:
            logging.error(f"Source {source} is invalid or inaccessible")

    for source in SHAREPOINT_URLS:
        if validate_source(source):
            assignments = assign_agent(source, is_sharepoint=True)
            for assignment in assignments:
                logging.info(f"Assigned {assignment.document_path} to {assignment.assigned_agent}")
            results.extend(assignments)
        else:
            logging.error(f"Source {source} is invalid or inaccessible")

    return results

# Define the DAG
with DAG(
    dag_id="document_router",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
):
    assignments = route_documents()

    # Example downstream tasks for each agent
    @task
    def process_file_parsing_agent(assignment: DocumentAssignment):
        if assignment.assigned_agent == "File Parsing Agent":
            logging.info(f"Processing {assignment.document_path} with File Parsing Agent")
            # Add logic to send to File Parsing Agent

    @task
    def process_email_agent(assignment: DocumentAssignment):
        if assignment.assigned_agent == "Email Agent":
            logging.info(f"Processing {assignment.document_path} with Email Agent")
            # Add logic to send to Email Agent

    @task
    def process_transcript_agent(assignment: DocumentAssignment):
        if assignment.assigned_agent == "Transcript Agent":
            logging.info(f"Processing {assignment.document_path} with Transcript Agent")
            # Add logic to send to Transcript Agent

    @task
    def process_image_processing_agent(assignment: DocumentAssignment):
        if assignment.assigned_agent == "Image Processing Agent":
            logging.info(f"Processing {assignment.document_path} with Image Processing Agent")
            # Add logic to send to Image Processing Agent

    # Expand tasks to process assignments
    process_file_parsing_agent.expand(assignment=assignments)
    process_email_agent.expand(assignment=assignments)
    process_transcript_agent.expand(assignment=assignments)
    process_image_processing_agent.expand(assignment=assignments)
