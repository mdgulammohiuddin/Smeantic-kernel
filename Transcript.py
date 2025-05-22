import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent
import logging

# Logging config
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY not set")

# Define assets folder path
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# Cleaning + parsing logic
def clean_and_parse_transcript(file_path: str) -> str:
    try:
        if not os.path.exists(file_path):
            return f"Error: File does not exist at {file_path}"
        
        elements = partition(filename=file_path)
        lines = [str(e).strip() for e in elements if str(e).strip()]
        cleaned_text = "\n".join(lines)
        return cleaned_text
    except Exception as e:
        return f"Parsing error: {str(e)}"

# Define Pydantic AI Agent
transcript_analysis_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[clean_and_parse_transcript],
    system_prompt="""
You are a transcript analysis agent.

Instructions:
1. Extract 'File Path' and 'User Query' from input.
2. Call `clean_and_parse_transcript` with the 'File Path' to get cleaned transcript content.
3. Analyze the transcript to extract a clear, accurate summary based on the 'User Query'.
4. Return only the structured and relevant answer without extra text like 'Here is your result...'.
"""
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="transcript_summary_dag",
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["transcript", "pydantic-ai", "summarization"]
)
def transcript_summary_dag():

    @task.agent(agent=transcript_analysis_agent)
    def summarize_transcript(file_path: str, query: str) -> str:
        return f"User Query: {query}\nFile Path: {file_path}"

    @task
    def print_summary(summary: str):
        print("=== Summary Start ===")
        print(summary)
        print("=== Summary End ===")
        return summary

    # Path to the transcript
    transcript_file_path = os.path.join(ASSETS_DIR, "transcript.txt")
    user_query = "What are the key points discussed in the final section?"

    # DAG flow
    summary_output = summarize_transcript(file_path=transcript_file_path, query=user_query)
    print_summary(summary_output)

# Instantiate DAG
dag_instance = transcript_summary_dag()
