import os
import time
import json
from datetime import datetime, timedelta, timezone
from airflow.decorators import dag, task
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from pydantic_ai import Agent as PydanticAIAgent
import logging
from typing import Dict, Any, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai

# Download required NLTK data (only runs if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

class AgentOutputMetricsStructure:
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    parse_time: Optional[float] = None
    parse_start: Optional[str] = None
    clean_time: Optional[float] = None
    clean_start: Optional[str] = None
    original_length: Optional[int] = None
    cleaned_length: Optional[int] = None
    error_message: Optional[str] = None

class AgentOutputStructure:
    content: str
    metrics: AgentOutputMetricsStructure

def parse_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Parse document and return content with metadata"""
    start_time_func = time.time()
    current_utc_time = datetime.now(timezone.utc).isoformat()
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}", {
                "error": f"File not found at {file_path}",
                "parse_time": time.time() - start_time_func,
                "parse_start": current_utc_time
            }
        elements = partition(filename=file_path)
        content = "\n\n".join([str(e) for e in elements])
        return content, {
            "parse_time": time.time() - start_time_func,
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1][1:].upper(),
            "parse_start": current_utc_time
        }
    except Exception as e:
        logger.error(f"Parsing error for {file_path}: {e}", exc_info=True)
        return f"Parsing error: {str(e)}", {
            "error": f"Parsing error: {str(e)}",
            "parse_time": time.time() - start_time_func,
            "parse_start": current_utc_time
        }

def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content using NLTK for stopword removal and additional cleaning steps, return with metrics"""
    start_time_func = time.time()
    current_utc_time = datetime.now(timezone.utc).isoformat()
    try:
        # Step 1: Remove timestamps (e.g., 12:34:56, 12:34, or HH:MM:SS.mmm formats)
        timestamp_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?(\.\d{1,3})?\b'
        content = re.sub(timestamp_pattern, '', raw_content)

        # Step 2: Remove special characters and punctuation, keep alphanumeric and spaces
        content = re.sub(r'[^\w\s]', '', content)

        # Step 3: Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        content = re.sub(r'\s+', ' ', content).strip()

        # Step 4: Tokenize and remove stopwords and filler words
        stop_words = set(stopwords.words('english'))
        # Define common filler words (extend as needed)
        filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'i mean', 'kind of', 'sort of', 'well', 'okay'
        }
        # Combine stopwords and filler words
        words_to_remove = stop_words.union(filler_words)

        # Tokenize the content
        tokens = word_tokenize(content.lower())  # Convert to lowercase for consistency
        filtered_tokens = [word for word in tokens if word not in words_to_remove]

        # Reconstruct cleaned content
        cleaned = ' '.join(filtered_tokens)

        return cleaned, {
            "clean_time": time.time() - start_time_func,
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned),
            "clean_start": current_utc_time
        }
    except Exception as e:
        logger.error(f"Cleaning error: {e}", exc_info=True)
        return f"Cleaning error: {str(e)}", {
            "error": f"Cleaning error: {str(e)}",
            "clean_time": time.time() - start_time_func,
            "clean_start": current_utc_time
        }

document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
You are a transcript processing system. Your goal is to process a document, clean its content,
and then analyze it based on a user query.
You MUST return your entire response as a single, valid JSON object (string).
Do NOT use Markdown formatting for the overall JSON structure.
The JSON object should have two top-level keys: "content" and "metrics".

The "content" key should contain a string with query-relevant information in bullet points (this string can contain Markdown).
If an error occurs, "content" should summarize the error.

The "metrics" key should contain an object with the following fields:
  "file_type": (string, from parse_document metadata, or null if error)
  "file_size": (integer, from parse_document metadata, or null if error)
  "parse_time": (float, duration from parse_document metadata, or null if error)
  "parse_start": (string, ISO datetime from parse_document metadata, or null if error)
  "clean_time": (float, duration from clean_content metadata, or null if error before cleaning)
  "clean_start": (string, ISO datetime from clean_content metadata, or null if error before cleaning)
  "original_length": (integer, from clean_content metadata, or null if error before cleaning)
  "cleaned_length": (integer, from clean_content metadata, or null if error before cleaning)
  "error_message": (string, describe any error that occurred during parsing or cleaning, or null if no errors)

Follow these steps strictly:
1. Use `parse_document` with the file path provided in the prompt. If it fails, populate "error_message" in metrics, put an error summary in "content", and provide nulls or available data for other metric fields. Do not proceed to step 2 if parsing fails critically.
2. If parsing is successful, use `clean_content` on the raw content. If it fails, populate "error_message", summarize in "content", and provide nulls or available data for other clean-related metric fields.
3. If cleaning is successful, analyze cleaned content against the user query provided in the prompt for "content".
4. Construct the final JSON object string as described above.

Tool usage sequence (if successful): parse_document -> clean_content -> analysis.
Ensure your output is ONLY the JSON string. Example of output format:
{
  "content": "- Action item 1\\n- Action item 2",
  "metrics": {
    "file_type": "DOCX",
    "file_size": 12345,
    "parse_time": 1.23,
    "parse_start": "2025-01-01T12:00:00Z",
    "clean_time": 0.45,
    "clean_start": "2025-01-01T12:00:02Z",
    "original_length": 1000,
    "cleaned_length": 800,
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
    dag_id="transcript_processor_embed",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["transcript", "ai-agent", "json-output"],
    params={
        "file_name": "meeting_transcript.docx",
        "user_query": "List all action items"
    }
)
def transcript_pipeline():
    @task
    def prepare_context(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = kwargs.get('params', {})
        file_name = params.get('file_name', "meeting_transcript.docx")
        logger.info(f"Received params: {params}, file_name: {file_name}")
        # Normalize file path
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            file_path = os.path.join(ASSETS_DIR, file_name)
        user_query = params.get('user_query', "List all action items")
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}, falling back to default")
            # Fallback to default file
            default_file = os.path.join(ASSETS_DIR, "meeting_transcript.docx")
            if os.path.exists(default_file):
                file_path = default_file
                logger.info(f"Using default file: {file_path}")
            else:
                file_path = None
                logger.error(f"Default file not found: {default_file}")
        logger.info(f"Prepared context: file_path={file_path}, user_query={user_query}")
        return {
            "file_path": file_path,
            "user_query": user_query,
            "process_start_time": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=document_agent)
    def process_document_get_json(context: Dict[str, Any]) -> str:
        """Agent task that processes the document and returns a JSON string."""
        try:
            file_path = context['file_path']
            logger.info(f"Processing file: {file_path}")
            if not file_path:
                logger.error("No valid file path provided")
                return json.dumps({
                    "content": "Error: No valid file path provided",
                    "metrics": {
                        "error_message": "No valid file path provided"
                    }
                })
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return json.dumps({
                    "content": f"Error: File not found at {file_path}",
                    "metrics": {
                        "error_message": f"File not found at {file_path}"
                    }
                })
            # Pass context as a prompt string
            prompt = f"""
            User Query: {context['user_query']}
            File Path: {file_path}
            Process Start Time (UTC): {context['process_start_time']}
            """
            result = document_agent.run_sync(prompt)
            logger.info(f"Agent output: {result.data}")
            return result.data  # Extract the JSON string from the AgentRunResult object
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
            result = json.loads(json_string)
            logger.info(f"Parsed agent output: {result}")
            return result
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
    def embed_output(parsed_result: Dict[str, Any]) -> list:
        """Generate embeddings for the content field using OpenAI's text-embedding-ada-002."""
        logger.info(f"Embedding input: {parsed_result}")
        content = parsed_result.get('content', '')
        if not content or "error" in content.lower():
            logger.warning(f"No valid content to embed: {content}")
            return []
        try:
            response = openai_client.embeddings.create(
                input=content,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding: length={len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
            return []

    @task
    def format_final_output(result_dict: Dict[str, Any], embedding: list, context: Dict[str, Any]) -> str:
        """Generate final output from the parsed dictionary and embedding."""
        agent_content = result_dict.get('content', 'No relevant information found or error in processing.')
        metrics_data = result_dict.get('metrics', {})
        file_size_kb_str = "N/A"
        file_size_bytes = metrics_data.get('file_size')
        if isinstance(file_size_bytes, (int, float)):
            file_size_kb_str = f"{file_size_bytes / 1024:.2f} KB"
        elif file_size_bytes is not None:
            file_size_kb_str = "Invalid Size Data"
        parse_time_str = f"{metrics_data.get('parse_time'):.2f}s" if metrics_data.get('parse_time') is not None else "N/A"
        clean_time_str = f"{metrics_data.get('clean_time'):.2f}s" if metrics_data.get('clean_time') is not None else "N/A"
        reduction_str = "N/A"
        original_len = metrics_data.get('original_length')
        cleaned_len = metrics_data.get('cleaned_length')
        if isinstance(original_len, int) and isinstance(cleaned_len, int):
            reduction_str = f"{original_len - cleaned_len} characters"
        embedding_str = f"Length: {len(embedding)}" if embedding else "No embedding generated"
        output = f"""
## Transcript Processing Report

### File Metadata
- Type: {metrics_data.get('file_type', 'N/A')}
- Size: {file_size_kb_str}
- Process Started (Overall): {context['process_start_time']}

### Processing Metrics
1. Parsing Stage:
   - Duration: {parse_time_str}
   - Started: {metrics_data.get('parse_start', 'N/A')}

2. Cleaning Stage:
   - Duration: {clean_time_str}
   - Started: {metrics_data.get('clean_start', 'N/A')}
   - Original Length: {metrics_data.get('original_length', 'N/A')} characters
   - Cleaned Length: {metrics_data.get('cleaned_length', 'N/A')} characters
   - Content Reduction: {reduction_str}

3. Embedding Stage:
   - Embedding: {embedding_str}

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
    agent_json_result = process_document_get_json(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    embedding_result = embed_output(parsed_agent_result)
    final_report = format_final_output(parsed_agent_result, embedding_result, context_data)
    print(final_report)

# Instantiate DAG
transcript_processor_dag = transcript_pipeline()
