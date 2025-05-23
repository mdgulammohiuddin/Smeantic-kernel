import os
import time
from datetime import datetime, timedelta
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
            "file_type": os.path.splitext(file_path)[1][1:].upper()
        }
    except Exception as e:
        return f"Parsing error: {str(e)}", {}

# --------- Tool 2: Content Cleaner ---------
def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content and return with metrics"""
    try:
        start_time = time.time()
        
        # Basic cleaning operations
        cleaned = "\n".join([
            line.strip() 
            for line in raw_content.split("\n") 
            if line.strip()
        ])
        
        return cleaned, {
            "clean_time": time.time() - start_time,
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned)
        }
    except Exception as e:
        return f"Cleaning error: {str(e)}", {}

# --------- Agent Configuration ---------
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
You are an advanced document processing system. Follow these steps:

1. Use parse_document with the file path to get raw content
2. Use clean_content on the raw output to refine it
3. Analyze cleaned content against the user query
4. Return structured response containing:
   - Query-relevant information
   - File metadata (type, size)
   - Processing times for each stage
   - Content length metrics

Format response in Markdown with sections for data and metrics
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
    dag_id="document_processing_pipeline",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["document", "ai-agent", "metrics"],
    params={
        "file_name": "meeting_transcript.docx",
        "user_query": "List all action items from the document"
    }
)
def document_pipeline():

    @task
    def prepare_context(**kwargs) -> Dict[str, Any]:
        """Collect initial context and validate file"""
        params = kwargs.get('params', {})
        file_name = params.get('file_name')
        user_query = params.get('user_query')
        
        file_path = os.path.join(ASSETS_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
            
        return {
            "file_path": file_path,
            "user_query": user_query,
            "process_start": datetime.utcnow().isoformat()
        }

    @task.agent(agent=document_agent)
    def process_document(context: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-controlled processing pipeline"""
        return f"""
        User Query: {context['user_query']}
        File Path: {context['file_path']}
        """

    @task
    def format_final_output(result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate final formatted output with all metrics"""
        metrics = result.get('metrics', {})
        content = result.get('content', '')
        
        output = f"""
## Document Processing Report

### File Metadata
- Type: {metrics.get('file_type', 'N/A')}
- Size: {metrics.get('file_size', 0)/1024:.2f} KB
- Source: {os.path.basename(context['file_path'])}

### Processing Metrics
- Total Time: {metrics.get('total_time', 0):.2f}s
  - Parsing: {metrics.get('parse_time', 0):.2f}s 
  - Cleaning: {metrics.get('clean_time', 0):.2f}s
- Content Length:
  - Original: {metrics.get('original_length', 0)} chars
  - Cleaned: {metrics.get('cleaned_length', 0)} chars

### Query Results
{content}

### Timeline
- Started: {context['process_start']}
- Completed: {datetime.utcnow().isoformat()}
        """
        print(output)
        return output

    # DAG execution flow
    context = prepare_context()
    agent_result = process_document(context)
    final_output = format_final_output(agent_result, context)

# Instantiate DAG
document_pipeline_dag = document_pipeline()
