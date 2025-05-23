import os
import time
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent
import logging
from typing import Dict, Any

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

# 1. Tool function from code 1
def parse_document_with_unstructured(file_url: str) -> str:
    """Modified from code 1 to handle different file types"""
    try:
        actual_file_path = file_url
        if not os.path.isabs(file_url) and not file_url.startswith(("http://", "https://")):
            path_in_assets = os.path.join(ASSETS_DIR, file_url)
            if os.path.exists(path_in_assets):
                actual_file_path = path_in_assets
            elif os.path.exists(file_url):
                actual_file_path = os.path.abspath(file_url)
            else:
                return f"Error: File not found. Checked paths related to: {file_url}"
        
        if not os.path.exists(actual_file_path) and not actual_file_path.startswith(("http://", "https://")):
             return f"Error: File not found at resolved path: {actual_file_path}"

        elements = partition(filename=actual_file_path)
        return "\n\n".join([str(element) for element in elements])
    except Exception as e:
        return f"An error occurred during parsing: {e}"

# 2. Define Pydantic AI agent with tool
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document_with_unstructured],
    system_prompt="""
You are a document processing expert. Process input formatted as:
User Query: [query]
File Path: [path]

1. Extract and validate file path
2. Use parse_document_with_unstructured to get content
3. Analyze content against query
4. Return structured response with:
   - Relevant information matching query
   - File metadata (type, size)
   - Processing timestamps
Format: Markdown with sections for data and metrics
"""
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

@dag(
    dag_id="enhanced_document_processor",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["document", "pydantic-ai", "metrics"],
    params={
        "file_name": "KB_0000268.pdf",
        "user_query": "What are the key points in this document?"
    }
)
def enhanced_document_dag():

    @task
    def get_file_metadata(file_name: str) -> Dict[str, Any]:
        """Collect initial file metadata"""
        file_path = os.path.join(ASSETS_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_name)[1].lstrip('.'),
            "start_time": datetime.utcnow().isoformat()
        }

    @task.agent(agent=document_agent)
    def parse_with_agent(metadata: Dict[str, Any], user_query: str) -> str:
        """Agent task for parsing and initial processing"""
        return f"User Query: {user_query}\nFile Path: {metadata['file_path']}"

    @task
    def process_results(parsed_output: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent output and collect metrics"""
        end_time = datetime.utcnow()
        return {
            "content": parsed_output,
            "metrics": {
                "file_size": metadata["file_size"],
                "file_type": metadata["file_type"],
                "processing_time": (datetime.fromisoformat(end_time) - 
                                  datetime.fromisoformat(metadata["start_time"])).total_seconds(),
                "stages": {
                    "parsing_start": metadata["start_time"],
                    "processing_end": end_time.isoformat()
                }
            }
        }

    @task
    def final_output(result: Dict[str, Any]) -> str:
        """Format final output with metrics"""
        output = f"""
## Document Processing Report

### File Information
- Type: {result['metrics']['file_type'].upper()}
- Size: {result['metrics']['file_size']/1024:.2f} KB

### Processing Metrics
- Total Time: {result['metrics']['processing_time']:.2f} seconds
- Parsing Started: {result['metrics']['stages']['parsing_start']}
- Processing Completed: {result['metrics']['stages']['processing_end']}

### Extracted Content
{result['content']}
        """
        print(output)
        return output

    # DAG execution flow
    file_name = "{{ params.file_name }}"
    user_query = "{{ params.user_query }}"
    
    metadata = get_file_metadata(file_name)
    parsed = parse_with_agent(metadata, user_query)
    processed = process_results(parsed, metadata)
    final_output(processed)

# Instantiate DAG
enhanced_dag = enhanced_document_dag()
