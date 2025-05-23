import os
import time
from datetime import datetime, timedelta, timezone
from airflow.decorators import dag, task
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent
import logging
from typing import Dict, Any, Tuple

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
            return f"Error: File not found at {file_path}", {}
        
        elements = partition(filename=file_path)
        content = "\n\n".join([str(e) for e in elements])
        
        return content, {
            "parse_time": time.time() - start_time,
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1][1:].upper(),
            "parse_start": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return f"Parsing error: {str(e)}", {}

# --------- Tool 2: Content Cleaner ---------
def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content and return with metrics"""
    try:
        start_time = time.time()
        clean_start = datetime.now(timezone.utc).isoformat()
        
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
            "clean_start": clean_start
        }
    except Exception as e:
        return f"Cleaning error: {str(e)}", {}

# --------- Agent Configuration ---------
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
You are a transcript processing system. Follow these steps strictly:

1. FIRST use parse_document with the provided file path to get raw content
2. THEN use clean_content on the raw output from step 1
3. FINALLY analyze the cleaned content against the user query
4. Return structured response containing:
   - Query-relevant information in bullet points
   - File metadata (type, size)
   - Processing times for each stage
   - Content length metrics

Tool usage sequence is MANDATORY:
parse_document -> clean_content -> analysis

NEVER repeat tools. Format response in Markdown.
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
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["transcript", "ai-agent", "metrics"],
    params={
        "file_name": "meeting_transcript.docx",
        "user_query": "List all action items from the document"
    }
)
def transcript_pipeline():

    @task
    def prepare_context(**kwargs) -> Dict[str, Any]:
        """Collect initial context with proper timezone"""
        params = kwargs.get('params', {})
        return {
            "file_path": os.path.join(ASSETS_DIR, params.get('file_name')),
            "user_query": params.get('user_query'),
            "process_start": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=document_agent)
    def process_document(context: Dict[str, Any]) -> Dict[str, Any]:
        """Agent task with enforced tool sequence"""
        return f"""
        User Query: {context['user_query']}
        File Path: {context['file_path']}
        """

    @task
    def format_final_output(result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate final output with timezone-aware timestamps"""
        metrics = result.get('metrics', {})
        output = f"""
## Transcript Processing Report

### File Metadata
- Type: {metrics.get('file_type', 'N/A')}
- Size: {metrics.get('file_size', 0)/1024:.2f} KB
- Process Started: {context['process_start']}

### Processing Metrics
1. Parsing Stage:
   - Duration: {metrics.get('parse_time', 0):.2f}s
   - Started: {metrics.get('parse_start', 'N/A')}

2. Cleaning Stage:
   - Duration: {metrics.get('clean_time', 0):.2f}s
   - Started: {metrics.get('clean_start', 'N/A')}
   - Content Reduction: {metrics.get('original_length', 0) - metrics.get('cleaned_length', 0)} characters

### Analysis Results
{result.get('content', 'No relevant information found')}

### Final Output
Generated at: {datetime.now(timezone.utc).isoformat()}
        """
        print(output)
        return output

    # DAG execution flow
    context = prepare_context()
    agent_result = process_document(context)
    final_output = format_final_output(agent_result, context)

# Instantiate DAG
transcript_processor = transcript_pipeline()
