import os
import json
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
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from datetime import datetime, timedelta, timezone

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Custom Logger Setup
CUSTOM_LOG_DIR_NAME = "dag_run_logs"
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_GUESS, "Backend")
CUSTOM_LOG_DIR = os.path.join(CUSTOM_LOG_BASE_DIR, CUSTOM_LOG_DIR_NAME)

custom_logger = logging.getLogger('TranscriptProcessorCustomLogger')
custom_logger.setLevel(logging.INFO)

def setup_custom_file_handler(logger_instance, log_file_name_prefix="transcript_processor"):
    if not logger_instance.handlers:
        try:
            if not os.path.exists(CUSTOM_LOG_DIR):
                os.makedirs(CUSTOM_LOG_DIR, exist_ok=True)
            log_file_path = os.path.join(CUSTOM_LOG_DIR, f"{log_file_name_prefix}_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(f"Custom file logger initialized. Logging to: {log_file_path}")
        except Exception as e:
            print(f"CRITICAL: Error setting up custom file logger: {e}")

setup_custom_file_handler(custom_logger)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('unstructured').setLevel(logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    custom_logger.error("OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

class AgentOutputMetricsStructure:
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    original_length: Optional[int] = None
    cleaned_length: Optional[int] = None
    error_message: Optional[str] = None

class AgentOutputStructure:
    content: str
    metrics: AgentOutputMetricsStructure

def parse_document(file_path: str, expected_file_path: str = None) -> Tuple[str, Dict[str, Any]]:
    """Parse document and return content with metadata, validate file path"""
    if expected_file_path and os.path.normpath(file_path) != os.path.normpath(expected_file_path):
        error_msg = f"Unauthorized file access attempted: {file_path}. Expected: {expected_file_path}"
        custom_logger.error(error_msg)
        return error_msg, {"error": error_msg}
    try:
        if not os.path.exists(file_path):
            error_msg = f"File not found at {file_path}"
            custom_logger.error(error_msg)
            return error_msg, {"error": error_msg}
        elements = partition(filename=file_path)
        content = "\n\n".join([str(e) for e in elements])
        custom_logger.info(f"Parsed document: {file_path}, length={len(content)}")
        return content, {
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1][1:].upper()
        }
    except Exception as e:
        error_msg = f"Parsing error for {file_path}: {e}"
        custom_logger.error(error_msg, exc_info=True)
        return error_msg, {"error": error_msg}

def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content using NLTK for stopword removal and additional cleaning steps"""
    try:
        timestamp_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?(\.\d{1,3})?\b'
        content = re.sub(timestamp_pattern, '', raw_content)
        content = re.sub(r'[^\w\s]', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        stop_words = set(stopwords.words('english'))
        filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'i mean', 'kind of', 'sort of', 'well', 'okay'
        }
        words_to_remove = stop_words.union(filler_words)
        tokens = word_tokenize(content.lower())
        filtered_tokens = [word for word in tokens if word not in words_to_remove]
        cleaned = ' '.join(filtered_tokens)
        custom_logger.info(f"Cleaned content: original_length={len(raw_content)}, cleaned_length={len(cleaned)}")
        return cleaned, {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned)
        }
    except Exception as e:
        error_msg = f"Cleaning error: {e}"
        custom_logger.error(error_msg, exc_info=True)
        return error_msg, {"error": error_msg}

document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
You are a transcript processing system. Your goal is to process EXACTLY ONE document specified in the prompt, clean its content, and analyze it based on the user query.
You MUST process ONLY the file path provided in the 'File Path' field of the prompt. Processing any other file is strictly forbidden and must result in an immediate error.
You MUST return a single, valid JSON object (string) for the specified file only, with no Markdown formatting in the JSON structure.
The JSON object must have two keys: "content" and "metrics".

The "content" key must contain a string with query-relevant information in bullet points (this string may use Markdown).
If an error occurs, "content" must summarize the error.

The "metrics" key must contain an object with:
  "file_type": (string, from parse_document metadata, or null if error)
  "file_size": (integer, from parse_document metadata, or null if error)
  "original_length": (integer, from clean_content metadata, or null if error before cleaning)
  "cleaned_length": (integer, from clean_content metadata, or null if error before cleaning)
  "error_message": (string, describe any error during parsing or cleaning, or null if no errors)

Follow these steps:
1. Extract and validate the file path from the 'File Path' field in the prompt. If the file path is missing or invalid, return an error JSON.
2. Call `parse_document` with the validated file path, passing the same file path as `expected_file_path`. If parsing fails, return an error JSON with appropriate metrics.
3. If parsing succeeds, call `clean_content` on the parsed content. If cleaning fails, return an error JSON with metrics.
4. If cleaning succeeds, analyze the cleaned content for the user query from the 'User Query' field and generate bullet-pointed content.
5. Return a JSON string with the content and metrics for the specified file only.

Tool sequence: parse_document -> clean_content -> analysis.
NEVER call `parse_document` for any file other than the one in the prompt's 'File Path' field.
If an unauthorized file path is detected, return an error JSON immediately.

Example output:
{
  "content": "- Action item 1\n- Action item 2",
  "metrics": {
    "file_type": "DOCX",
    "file_size": 12345,
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
    "execution_timeout": timedelta(minutes=15),
}

@dag(
    dag_id="transcript_processor_embed",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["transcript", "ai-agent", "json-output", "embedding", "faiss"],
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
        custom_logger.info(f"Received params: {params}, file_name: {file_name}")
        if not file_name or (isinstance(file_name, str) and file_name.startswith("{{")):
            error_msg = f"Invalid file_name: '{file_name}'"
            custom_logger.error(error_msg)
            raise ValueError(error_msg)
        candidate_path = os.path.join(ASSETS_DIR, file_name) if not os.path.isabs(file_name) else file_name
        if os.path.exists(candidate_path):
            custom_logger.info(f"File found at: {candidate_path}")
            file_path = candidate_path
        else:
            fallback_path = os.path.join(ASSETS_DIR, os.path.basename(file_name))
            if os.path.exists(fallback_path):
                custom_logger.info(f"Main path '{candidate_path}' not found. Using fallback: '{fallback_path}'")
                file_path = fallback_path
            else:
                error_msg = f"File not found. Input: '{file_name}', Checked: '{candidate_path}', Fallback: '{fallback_path}'"
                custom_logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        user_query = params.get('user_query', "List all action items")
        custom_logger.info(f"Prepared context: file_path={file_path}, user_query={user_query}")
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
            user_query = context['user_query']
            process_start_time = context['process_start_time']
            custom_logger.info(f"Processing file: {file_path}")
            if not file_path:
                error_msg = "No valid file path provided"
                custom_logger.error(error_msg)
                return json.dumps({
                    "content": error_msg,
                    "metrics": {"error_message": error_msg}
                })
            if not os.path.exists(file_path):
                error_msg = f"File not found at {file_path}"
                custom_logger.error(error_msg)
                return json.dumps({
                    "content": error_msg,
                    "metrics": {"error_message": error_msg}
                })
            prompt = f"""
User Query: {user_query}
File Path: {file_path}
Process Start Time (UTC): {process_start_time}
"""
            custom_logger.info(f"Agent prompt: {prompt}")
            try:
                os.environ['EXPECTED_FILE_PATH'] = file_path
                result = document_agent.run_sync(prompt)
                custom_logger.info(f"Agent output: {result.data}")
                os.environ.pop('EXPECTED_FILE_PATH', None)
                return result.data
            except Exception as llm_error:
                error_msg = f"LLM processing error: {str(llm_error)}"
                custom_logger.error(error_msg, exc_info=True)
                return json.dumps({
                    "content": error_msg,
                    "metrics": {"error_message": error_msg}
                })
        except Exception as e:
            error_msg = f"Agent processing error: {str(e)}"
            custom_logger.error(error_msg, exc_info=True)
            return json.dumps({
                "content": error_msg,
                "metrics": {"error_message": error_msg}
            })

    @task
    def parse_agent_json_output(json_string: str) -> Dict[str, Any]:
        """Parses the JSON string output from the agent into a dictionary."""
        try:
            result = json.loads(json_string)
            custom_logger.info(f"Parsed agent output: {result}")
            if result.get('metrics', {}).get('error_message'):
                custom_logger.warning(f"Agent returned error: {result['metrics']['error_message']}")
            return result
        except json.JSONDecodeError as e:
            error_msg = f"JSONDecodeError: {e}. Received: {json_string[:200]}..."
            custom_logger.error(error_msg)
            return {
                "content": "Error: Agent returned malformed JSON.",
                "metrics": {"error_message": error_msg}
            }

    @task
    def create_embeddings(parsed_result: Dict[str, Any]) -> list:
        """Generate embeddings for the content field using OpenAI's text-embedding-ada-002."""
        custom_logger.info(f"Embedding input: {parsed_result}")
        content = parsed_result.get('content', '')
        if not content or "error" in content.lower():
            custom_logger.warning(f"No valid content to embed: {content}")
            return []
        try:
            embeddings_client = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_vector = embeddings_client.embed_query(content)
            custom_logger.info(f"Generated embedding: length={len(embedding_vector)}")
            return embedding_vector
        except Exception as e:
            custom_logger.error(f"Embedding error: {e}", exc_info=True)
            return []

    @task
    def store_embeddings_in_faiss(original_text: Dict[str, Any], embedding_vector: list) -> None:
        """Stores the provided text and its pre-computed embedding in an in-memory FAISS index."""
        content = original_text.get('content', '')
        if not embedding_vector or not content:
            custom_logger.warning(f"Empty embedding vector or content. Skipping FAISS storage: content={content[:100]}...")
            return
        custom_logger.info(f"Storing embedding in FAISS. Text (first 100 chars): '{content[:100]}...'")
        try:
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_dim = len(embedding_vector)
            if embedding_dim == 0:
                custom_logger.error("Embedding vector dimension is 0. Cannot create FAISS index.")
                raise ValueError("Cannot initialize FAISS with 0-dimension embedding.")
            index = faiss.IndexFlatL2(embedding_dim)
            docstore = InMemoryDocstore()
            index_to_docstore_id = {}
            vector_store = FAISS(
                embedding_function=embeddings_model,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            vector_store.add_embeddings(text_embeddings=[(content, embedding_vector)])
            custom_logger.info(f"Successfully added text and embedding to FAISS. Index has {vector_store.index.ntotal} entries.")
        except Exception as e:
            custom_logger.error(f"FAISS storage error: {e}", exc_info=True)
            raise

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
- Process Started: {context['process_start_time']}

### Processing Metrics
- Original Length: {metrics_data.get('original_length', 'N/A')} characters
- Cleaned Length: {metrics_data.get('cleaned_length', 'N/A')} characters
- Content Reduction: {reduction_str}
- Embedding: {embedding_str}

{f"### Errors during Processing\n- {metrics_data.get('error_message')}" if metrics_data.get('error_message') else "No errors reported by agent."}

### Analysis Results
{agent_content}

### Final Output
Generated at: {datetime.now(timezone.utc).isoformat()}
        """
        print(output)
        custom_logger.info(f"Final report generated: {output[:200]}...")
        return output

    # DAG execution flow
    context_data = prepare_context()
    agent_json_result = process_document_get_json(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    embedding_result = create_embeddings(parsed_agent_result)
    store_embeddings_in_faiss(parsed_agent_result, embedding_result)
    final_report = format_final_output(parsed_agent_result, embedding_result, context_data)
    print(final_report)

# Instantiate DAG
transcript_processor_dag = transcript_pipeline()
