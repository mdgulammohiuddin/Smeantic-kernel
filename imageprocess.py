import os
import time
from datetime import datetime, timedelta, timezone
import json
from airflow.decorators import dag, task
from dotenv import load_dotenv
import easyocr
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from pydantic_ai import Agent as PydanticAIAgent
import logging
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

class AgentOutputMetricsStructure:
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    parse_time: Optional[float] = None
    parse_start: Optional[str] = None
    error_message: Optional[str] = None

class AgentOutputStructure:
    content: str
    metrics: AgentOutputMetricsStructure

def extract_text_from_image(image_source: str) -> Tuple[str, Dict[str, Any]]:
    """Extracts OCR text from the provided image source and returns with metadata."""
    start_time_func = time.time()
    current_utc_time = datetime.now(timezone.utc).isoformat()
    try:
        # Load the image
        if image_source.startswith(("http://", "https://")):
            response = requests.get(image_source)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            file_size = len(response.content)
            file_type = "URL"
        else:
            if not os.path.exists(image_source):
                return f"Error: Image not found at {image_source}", {
                    "error": f"Image not found at {image_source}",
                    "parse_time": time.time() - start_time_func,
                    "parse_start": current_utc_time
                }
            image = Image.open(image_source)
            file_size = os.path.getsize(image_source)
            file_type = os.path.splitext(image_source)[1][1:].upper()

        # Convert image to numpy array
        image_np = np.array(image)

        # Perform OCR
        ocr_data = reader.readtext(image_np)
        ocr_text = " ".join([text[1] for text in ocr_data])

        return ocr_text if ocr_text else "No text extracted", {
            "parse_time": time.time() - start_time_func,
            "file_size": file_size,
            "file_type": file_type,
            "parse_start": current_utc_time
        }
    except Exception as e:
        logger.error(f"OCR extraction error for {image_source}: {e}", exc_info=True)
        return f"OCR extraction error: {str(e)}", {
            "error": f"OCR extraction error: {str(e)}",
            "parse_time": time.time() - start_time_func,
            "parse_start": current_utc_time
        }

image_agent = PydanticAIAgent(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    tools=[extract_text_from_image],
    system_prompt="""
You are an image text processing system. Your goal is to extract text from an image and analyze it based on a user query.
You MUST return your entire response as a single, valid JSON object (string).
Do NOT use Markdown formatting for the overall JSON structure.
The JSON object should have two top-level keys: "content" and "metrics".

The "content" key should contain a string with query-relevant information in bullet points (this string can contain Markdown).
If an error occurs, "content" should summarize the error.

The "metrics" key should contain an object with the following fields:
  "file_type": (string, from extract_text_from_image metadata, or null if error)
  "file_size": (integer, from extract_text_from_image metadata, or null if error)
  "parse_time": (float, duration from extract_text_from_image metadata, or null if error)
  "parse_start": (string, ISO datetime from extract_text_from_image metadata, or null if error)
  "error_message": (string, describe any error that occurred during extraction, or null if no errors)

Follow these steps strictly:
1. Use `extract_text_from_image` with the image path. If it fails, populate "error_message" in metrics, put an error summary in "content", and provide nulls or available data for other metric fields.
2. If extraction is successful, analyze the extracted text against the user query for "content".
3. Construct the final JSON object string as described above.

Tool usage sequence: extract_text_from_image -> analysis.
Ensure your output is ONLY the JSON string. Example of output format:
{
  "content": "- Invoice number: INV123\\n- Date: 2025-01-01",
  "metrics": {
    "file_type": "PNG",
    "file_size": 12345,
    "parse_time": 1.23,
    "parse_start": "2025-01-01T12:00:00Z",
    "error_message": null
  }
}
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
    dag_id="image_processor",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["image", "ocr", "ai-agent", "json-output"],
    params={
        "image_file": "invoice.png",
        "user_query": "Extract key details from the image"
    }
)
def image_processing_pipeline():
    @task
    def prepare_context(**kwargs) -> Dict[str, Any]:
        params = kwargs.get('params', {})
        image_file = params.get('image_file', "invoice.png")
        user_query = params.get('user_query', "Extract key details from the image")
        return {
            "image_path": os.path.join(ASSETS_DIR, image_file),
            "user_query": user_query,
            "process_start": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=image_agent)
    def process_image_get_json(context: Dict[str, Any]) -> str:
        """Agent task that processes the image and returns a JSON string."""
        try:
            result = image_agent.run(
                query=context['user_query'],
                image_path=context['image_path']
            )
            return result
        except Exception as e:
            logger.error(f"Agent processing error: {e}", exc_info=True)
            return json.dumps({
                "content": f"Agent processing error: {str(e)}",
                "metrics": {
                    "error_message": f"Agent processing error: {str(e)}"
                }
            })

    @task
    def parse_agent_json_output(json_string: str) -> Dict[str, Any]:
        """Parses the JSON string output from the agent into a dictionary."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from agent: {e}")
            logger.error(f"Received string: {json_string}")
            return {
                "content": "Error: Agent returned malformed JSON.",
                "metrics": {
                    "error_message": f"JSONDecodeError: {e}. Received: {json_string[:200]}..."
                }
            }

    @task
    def format_final_output(result_dict: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate final output from the parsed dictionary."""
        agent_content = result_dict.get('content', 'No relevant information found or error in processing.')
        metrics_data = result_dict.get('metrics', {})
        file_size_kb_str = "N/A"
        file_size_bytes = metrics_data.get('file_size')
        if isinstance(file_size_bytes, (int, float)):
            file_size_kb_str = f"{file_size_bytes / 1024:.2f} KB"
        elif file_size_bytes is not None:
            file_size_kb_str = "Invalid Size Data"
        parse_time_str = f"{metrics_data.get('parse_time'):.2f}s" if metrics_data.get('parse_time') is not None else "N/A"
        output = f"""
## Image Processing Report

### File Metadata
- Type: {metrics_data.get('file_type', 'N/A')}
- Size: {file_size_kb_str}
- Process Started (Overall): {context['process_start']}

### Processing Metrics
1. OCR Extraction Stage:
   - Duration: {parse_time_str}
   - Started: {metrics_data.get('parse_start', 'N/A')}

{f"### Errors during Processing\n- {metrics_data.get('error_message')}" if metrics_data.get('error_message') else "No errors reported by agent."}

### Analysis Results
{agent_content}

### Final Output
Generated at: {datetime.now(timezone.utc).isoformat()}
        """
        print(output)
        return output

    # DAG execution flow
    context_data = prepare_context()
    agent_json_result = process_image_get_json(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    final_report = format_final_output(parsed_agent_result, context_data)
    print(final_report)

# Instantiate DAG
image_processor_dag = image_processing_pipeline()
