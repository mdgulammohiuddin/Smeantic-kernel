import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define assets directory
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# 1. Function to parse transcript (returns list of elements)
def parse_file_elements(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return partition(filename=file_path)

# 2. Function to clean transcript text from parsed elements
def clean_elements(elements: list) -> str:
    return "\n".join([str(e).strip() for e in elements if str(e).strip()])

# Define Pydantic AI agent
transcript_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[],
    system_prompt="""
You are a transcript summarization agent.

You will receive a cleaned transcript and a user query. Your task is to analyze the transcript
and return only the relevant, structured answer related to the query.
Avoid any conversational language.
"""
)

# DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

@dag(
    dag_id="split_parse_clean_transcript_query_dag",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["transcript", "split-clean", "pydantic-ai"],
    params={
        "file_name": "meeting_transcript.docx",
        "user_query": "List all decisions made in the meeting."
    }
)
def transcript_dag():

    @task
    def parse_transcript(file_name: str):
        file_path = os.path.join(ASSETS_DIR, file_name)
        logger.info(f"Parsing: {file_path}")
        return parse_file_elements(file_path)

    @task
    def clean_transcript(elements):
        return clean_elements(elements)

    @task.agent(agent=transcript_agent)
    def summarize_transcript(cleaned_text: str, user_query: str) -> str:
        return f"Transcript:\n{cleaned_text}\n\nQuery:\n{user_query}"

    @task
    def print_summary(result: str):
        print("=== Final Summary ===")
        print(result)
        return result

    # Read UI params
    file_name = "{{ params.file_name }}"
    user_query = "{{ params.user_query }}"

    # DAG execution flow
    raw_elements = parse_transcript(file_name)
    cleaned = clean_transcript(raw_elements)
    summary = summarize_transcript(cleaned_text=cleaned, user_query=user_query)
    print_summary(summary)

# Instantiate DAG
dag_instance = transcript_dag()
