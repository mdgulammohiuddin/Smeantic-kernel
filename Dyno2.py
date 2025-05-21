from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Standard logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)

# Pydantic AI Agent base import
from pydantic_ai import Agent as PydanticAIAgent

# Tool functions (stubs) for each domain; replace with real implementations
def parse_document_with_unstructured(file_url: str) -> str:
    return f"Parsed document content for {file_url}"

def process_images_in_document(file_url: str) -> str:
    return f"Processed images in {file_url}"

def parse_email(file_url: str) -> str:
    return f"Extracted email metadata & body for {file_url}"

def parse_transcript(file_url: str) -> str:
    return f"Transcribed and summarized {file_url}"

# Instantiate one Pydantic AI agent per workload:
file_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document_with_unstructured],
    system_prompt="You are a document parsing assistant. Use the provided tool to parse any PDF, PPTX, XLSX."
)

image_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[process_images_in_document],
    system_prompt="You are an image processing assistant. Extract and interpret images from documents."
)

email_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_email],
    system_prompt="You are an email assistant. Parse .eml or .msg files and extract headers, body, and attachments."
)

transcript_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_transcript],
    system_prompt="You are a transcript assistant. Parse .vtt, .txt, or .docx transcripts and summarize key points."
)

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    default_args=default_args,
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['pydantic-ai', 'routing', 'multi-agent']
)
def multi_agent_routing_dag():

    @task
    def route_to_agent(input_path: str) -> dict:
        """
        Inspect the input_path extension (and document contents for images)
        and returns a dict with 'agent' and 'payload' ready for the next task.
        """
        ext = os.path.splitext(input_path)[1].lower()

        # Document types
        if ext in {'.pdf', '.pptx', '.ppt', '.xlsx', '.docx'}:
            # Optionally detect embedded images
            has_images = False
            # [Here you might call partition() or similar to peek for images...]
            # For demo, we'll assume PPTX always has images
            if ext in {'.pptx', '.ppt'}:
                has_images = True

            if has_images:
                return {'agent': 'image', 'path': input_path}

            return {'agent': 'file', 'path': input_path}

        # Email types
        if ext in {'.eml', '.msg'}:
            return {'agent': 'email', 'path': input_path}

        # Transcript types
        if ext in {'.vtt', '.txt', '.docx'}:
            return {'agent': 'transcript', 'path': input_path}

        # Fallback
        return {'agent': 'file', 'path': input_path}

    @task.agent(agent=file_agent)
    def run_file_agent(path: str) -> str:
        return path

    @task.agent(agent=image_agent)
    def run_image_agent(path: str) -> str:
        return path

    @task.agent(agent=email_agent)
    def run_email_agent(path: str) -> str:
        return path

    @task.agent(agent=transcript_agent)
    def run_transcript_agent(path: str) -> str:
        return path

    @task
    def collect_results(routed: dict,
                        file_res: str = None,
                        image_res: str = None,
                        email_res: str = None,
                        transcript_res: str = None) -> str:
        """
        Consolidate the output from whichever agent ran.
        """
        agent = routed['agent']
        if agent == 'file':
            result = file_res
        elif agent == 'image':
            result = image_res
        elif agent == 'email':
            result = email_res
        else:
            result = transcript_res

        return {
            'used_agent': agent,
            'output': result
        }

    # DAG wiring:
    # Replace this path with an actual incoming file
    incoming = "/path/to/input/document.pdf"
    routed = route_to_agent(incoming)

    # Fan‐out all possible agent tasks, Airflow will only run the one matching the route:
    file_out = run_file_agent(path=routed['path'])
    img_out  = run_image_agent(path=routed['path'])
    email_out = run_email_agent(path=routed['path'])
    trans_out = run_transcript_agent(path=routed['path'])

    final = collect_results(
        routed=routed,
        file_res=file_out,
        image_res=img_out,
        email_res=email_out,
        transcript_res=trans_out
    )

    return final

# Instantiate the DAG
multi_agent_routing_dag = multi_agent_routing_dag()
      
