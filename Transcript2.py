import os
import time
from datetime import datetime, timedelta, timezone
from airflow.decorators import dag, task
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent # Ensure this is the correct import
# If pydantic-ai is a different package, adjust the import.
# Assuming it's from `instructor` or a similar library that uses Pydantic with LLMs,
# the agent setup might be slightly different, but the core idea of using an output model remains.
# For this example, I'll assume PydanticAIAgent is correctly named and used.

import logging
from typing import Dict, Any, Tuple, Optional

# Import Pydantic BaseModel and Field
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# --------- Tool 1: Document Parser ---------
def parse_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Parse document and return content with metadata"""
    try:
        start_time = time.time()
        
        if not os.path.exists(file_path):
            # Return error information in a structured way if possible, or raise an exception
            # For simplicity, tools here return error strings, but agent needs to handle them.
            return f"Error: File not found at {file_path}", {
                "error": f"File not found at {file_path}",
                "parse_time": time.time() - start_time,
                "parse_start": datetime.now(timezone.utc).isoformat()
            }
        
        elements = partition(filename=file_path)
        content = "\n\n".join([str(e) for e in elements])
        
        return content, {
            "parse_time": time.time() - start_time,
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1][1:].upper(),
            "parse_start": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return f"Parsing error: {str(e)}", {
            "error": f"Parsing error: {str(e)}",
            "parse_time": time.time() - start_time if 'start_time' in locals() else 0,
            "parse_start": datetime.now(timezone.utc).isoformat()
        }

# --------- Tool 2: Content Cleaner ---------
def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content and return with metrics"""
    try:
        start_time = time.time()
        clean_start_time = datetime.now(timezone.utc).isoformat()
        
        # Advanced cleaning operations
        cleaned = "\n".join([
            line.strip() 
            for line in raw_content.split("\n") 
            if line.strip() and not line.startswith(("Â", "�"))  # Remove special chars
        ])
        
        return cleaned, {
            "clean_time": time.time() - start_time,
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "clean_start": clean_start_time
        }
    except Exception as e:
        return f"Cleaning error: {str(e)}", {
             "error": f"Cleaning error: {str(e)}",
             "clean_time": time.time() - start_time if 'start_time' in locals() else 0,
             "clean_start": datetime.now(timezone.utc).isoformat()
        }

# --------- Pydantic Models for Agent Output ---------
class AgentOutputMetrics(BaseModel):
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    parse_time: Optional[float] = None
    parse_start: Optional[str] = None
    clean_time: Optional[float] = None
    clean_start: Optional[str] = None
    original_length: Optional[int] = None
    cleaned_length: Optional[int] = None
    error_message: Optional[str] = None # To capture any errors from tools

class AgentOutput(BaseModel):
    content: str = Field(description="Query-relevant information in bullet points (Markdown formatted). If an error occurred, summarize the error here.")
    metrics: AgentOutputMetrics

# --------- Agent Configuration ---------
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    output_model=AgentOutput, # Specify the output model
    system_prompt="""
You are a transcript processing system. Your goal is to process a document, clean its content,
and then analyze it based on a user query. You must populate the 'AgentOutput' model.

Follow these steps strictly:

1.  Use the `parse_document` tool with the provided file path. This tool returns the raw content
    and a dictionary of metadata (parse_time, file_size, file_type, parse_start).
    If `parse_document` returns an error in its content or metadata (e.g., file not found),
    capture this error in the `AgentOutput.metrics.error_message` field and summarize in `AgentOutput.content`.
    Do not proceed to step 2 if parsing fails critically.

2.  If parsing is successful, use the `clean_content` tool on the raw content from step 1.
    This tool returns the cleaned content and a dictionary of metadata (clean_time, original_length, cleaned_length, clean_start).
    If `clean_content` returns an error, capture this in `AgentOutput.metrics.error_message` and summarize in `AgentOutput.content`.

3.  If cleaning is successful, analyze the cleaned content against the user query to produce query-relevant information.
    This information should be formatted as Markdown bullet points.

4.  Populate all fields of the 'AgentOutput' Pydantic model:
    -   `AgentOutput.content`: Store the Markdown bullet points of query-relevant information. If an error occurred in prior steps, provide a summary of the error.
    -   `AgentOutput.metrics.file_type`: From `parse_document` metadata.
    -   `AgentOutput.metrics.file_size`: From `parse_document` metadata.
    -   `AgentOutput.metrics.parse_time`: From `parse_document` metadata.
    -   `AgentOutput.metrics.parse_start`: From `parse_document` metadata.
    -   `AgentOutput.metrics.clean_time`: From `clean_content` metadata.
    -   `AgentOutput.metrics.clean_start`: From `clean_content` metadata.
    -   `AgentOutput.metrics.original_length`: From `clean_content` metadata.
    -   `AgentOutput.metrics.cleaned_length`: From `clean_content` metadata.
    -   `AgentOutput.metrics.error_message`: Populate if any tool reported an error or if a step could not be completed.

Tool usage sequence is MANDATORY if preceding steps are successful:
`parse_document` -> `clean_content` -> analysis.

NEVER repeat tools. Ensure your final output is a valid instance of the 'AgentOutput' model.
If `parse_document` fails, try to populate `parse_time`, `parse_start` and the error in metrics, and a relevant message in content.
"""
)

# --------- DAG Definition ---------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

@dag(
    dag_id="transcript_processor",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), # Ensure start_date is in the past for immediate runs if schedule is None, or a future date for scheduled runs.
    catchup=False,
    tags=["transcript", "ai-agent", "metrics"],
    params={
        "file_name": "meeting_transcript.docx", # Ensure this file exists in ASSETS_DIR for testing
        "user_query": "List all action items from the document"
    }
)
def transcript_pipeline():

    @task
    def prepare_context(**kwargs) -> Dict[str, Any]:
        """Collect initial context with proper timezone"""
        params = kwargs.get('params', {})
        file_name = params.get('file_name', "meeting_transcript.docx") # Default for safety
        user_query = params.get('user_query', "List all action items.") # Default for safety
        
        # Ensure ASSETS_DIR is correctly resolved if DAG is in a subfolder
        # This __file__ might point to the DAG file's location.
        # If ASSETS_DIR is relative to the project root, you might need a more robust path construction.
        # For example, using `os.path.dirname(os.path.dirname(__file__))` if DAGs are in a 'dags' subdir.
        # The provided ASSETS_DIR calculation seems okay if this file is in a 'dags' folder and 'assets' is a sibling.
        
        return {
            "file_path": os.path.join(ASSETS_DIR, file_name),
            "user_query": user_query,
            "process_start": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=document_agent) # Type hint implicitly handled by Pydantic model if agent returns it
    def process_document(context: Dict[str, Any]) -> Dict[str, Any]: # Output should be Dict from Pydantic model dump
        """Agent task with enforced tool sequence.
        The input to the agent is constructed here.
        The PydanticAIAgent (with output_model) should return a Pydantic model instance.
        Airflow's @task.agent decorator or XCom serialization should handle converting this
        Pydantic model instance into a dictionary.
        """
        # The string returned here is the initial prompt/query for the agent's run method.
        return f"""
        User Query: {context['user_query']}
        File Path: {context['file_path']}
        Process Start Time (UTC): {context['process_start']}
        """

    @task
    def format_final_output(result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate final output with timezone-aware timestamps"""
        # 'result' is now expected to be a dictionary derived from the AgentOutput Pydantic model
        agent_content = result.get('content', 'No relevant information found or error in processing.')
        metrics_data = result.get('metrics', {})
        
        # Handle potential division by zero if file_size is None or 0
        file_size_kb = "N/A"
        if metrics_data.get('file_size') is not None:
            try:
                file_size_kb_val = float(metrics_data['file_size']) / 1024
                file_size_kb = f"{file_size_kb_val:.2f} KB"
            except TypeError: # Handles if file_size is not a number
                 file_size_kb = "Invalid Size"


        output = f"""
## Transcript Processing Report

### File Metadata
- Type: {metrics_data.get('file_type', 'N/A')}
- Size: {file_size_kb}
- Process Started (Overall): {context['process_start']}

### Processing Metrics
1. Parsing Stage:
   - Duration: {metrics_data.get('parse_time', 'N/A'):.2f}s
   - Started: {metrics_data.get('parse_start', 'N/A')}

2. Cleaning Stage:
   - Duration: {metrics_data.get('clean_time', 'N/A'):.2f}s
   - Started: {metrics_data.get('clean_start', 'N/A')}
   - Original Length: {metrics_data.get('original_length', 'N/A')} characters
   - Cleaned Length: {metrics_data.get('cleaned_length', 'N/A')} characters
   - Content Reduction: {(metrics_data.get('original_length', 0) or 0) - (metrics_data.get('cleaned_length', 0) or 0)} characters

{f"### Errors during Processing
- {metrics_data.get('error_message')}" if metrics_data.get('error_message') else ""}

### Analysis Results
{agent_content}

### Final Output
Generated at: {datetime.now(timezone.utc).isoformat()}
        """
        print(output)
        return output

    # DAG execution flow
    context_data = prepare_context()
    agent_result_data = process_document(context_data)
    final_report = format_final_output(agent_result_data, context_data)

# Instantiate DAG
transcript_processor_dag = transcript_pipeline()
