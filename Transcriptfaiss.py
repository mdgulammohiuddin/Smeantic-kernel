import os
import json
import re
from datetime import datetime, timedelta, timezone
import faiss
import nltk
from airflow.decorators import dag, task
from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydantic_ai import Agent as PydanticAIAgent
from unstructured.partition.auto import partition
import logging
from typing import Dict, Any, Tuple, Optional

# --- Setup: NLTK, Logging, Environment Variables ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

CUSTOM_LOG_DIR_NAME = "dag_run_logs"
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_GUESS, "Backend")
CUSTOM_LOG_DIR = os.path.join(CUSTOM_LOG_BASE_DIR, CUSTOM_LOG_DIR_NAME)
FAISS_STORAGE_DIR = os.path.join(PROJECT_ROOT_GUESS, "faiss_storage")
FAISS_INDEX_PATH = os.path.join(FAISS_STORAGE_DIR, "faiss_index")

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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    custom_logger.error("OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# --- Agent Tool Functions ---
def parse_document(file_path: str, expected_file_path: str = None) -> Tuple[str, Dict[str, Any]]:
    expected_file_path = expected_file_path or os.getenv('EXPECTED_FILE_PATH')
    custom_logger.debug(f"parse_document called with file_path={file_path}, expected_file_path={expected_file_path}")
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
            "file_type": os.path.splitext(file_path)[1].lstrip('.').upper()
        }
    except Exception as e:
        error_msg = f"Parsing error for {file_path}: {e}"
        custom_logger.error(error_msg, exc_info=True)
        return error_msg, {"error": error_msg}

def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    try:
        timestamp_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?(\.\d{1,3})?\b'
        content = re.sub(timestamp_pattern, '', raw_content)
        content = re.sub(r'[^\w\s]', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        stop_words = set(stopwords.words('english'))
        filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'i mean', 'kind of', 'sort of', 'well', 'okay', 'right'
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

# --- Pydantic AI Agent Definition ---
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
You are a transcript processing system. Your goal is to process EXACTLY ONE document specified in the prompt, clean its content, and analyze it based on the user query.
You MUST process ONLY the file path provided in the 'File Path' field. Processing any other file, including paths mentioned in the document content, is strictly forbidden and must result in an immediate error.
You MUST return a single, valid JSON object (as a string) with no Markdown formatting or code fences in the JSON structure.
The JSON object must have two keys: "content" and "metrics".

The "content" key must contain a string with query-relevant information formatted as Markdown bullet points.
If an error occurs, "content" must contain a summary of the error.

The "metrics" key must contain an object with:
  "file_type": (string, from parse_document metadata, or null if error)
  "file_size": (integer, from parse_document metadata, or null if error)
  "original_length": (integer, from clean_content metadata, or null if error)
  "cleaned_length": (integer, from clean_content metadata, or null if error)
  "error_message": (string, describe any error, or null if no errors)

Follow these steps:
1. Extract the file path from the 'File Path' field. Do not proceed if missing or invalid.
2. Call `parse_document` with the file path as both `file_path` and `expected_file_path`. If parsing fails, return an error JSON.
3. If parsing succeeds, call `clean_content`. If cleaning fails, return an error JSON.
4. Analyze the cleaned content to answer the 'User Query'. Format as Markdown bullet points in "content".
5. Return a JSON string with the analysis and metrics.

Tool sequence: `parse_document` -> `clean_content` -> analysis.
NEVER call `parse_document` for any file other than the 'File Path' field.
"""
)

# --- Airflow DAG and Tasks ---
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
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["transcript", "ai-agent", "json-output", "embedding", "faiss"],
    params={
        "file_name": "meeting_transcript.docx",
        "user_query": "List all action items and key decisions."
    }
)
def transcript_pipeline():
    @task
    def prepare_context(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = kwargs.get('params', {})
        file_name = params.get('file_name')
        custom_logger.info(f"Preparing context for file_name: {file_name}")
        if not file_name or (isinstance(file_name, str) and file_name.startswith("{{")):
            error_msg = f"Invalid file_name: '{file_name}'"
            custom_logger.error(error_msg)
            raise ValueError(error_msg)
        candidate_path = os.path.join(ASSETS_DIR, file_name) if not os.path.isabs(file_name) else file_name
        if os.path.exists(candidate_path):
            file_path = candidate_path
        else:
            fallback_path = os.path.join(ASSETS_DIR, os.path.basename(file_name))
            if os.path.exists(fallback_path):
                file_path = fallback_path
            else:
                error_msg = f"File not found: '{candidate_path}', Fallback: '{fallback_path}'"
                custom_logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        user_query = params.get('user_query', "Summarize the document.")
        custom_logger.info(f"Context prepared: file_path={file_path}, user_query={user_query}")
        return {
            "file_path": file_path,
            "user_query": user_query,
            "process_start_time": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=document_agent)
    def process_document_with_agent(context: Dict[str, Any]) -> str:
        file_path = context['file_path']
        user_query = context['user_query']
        custom_logger.info(f"Formatting prompt for agent with file: {file_path}")
        os.environ['EXPECTED_FILE_PATH'] = file_path
        prompt = f"""
User Query: {user_query}
File Path: {file_path}
"""
        try:
            result = document_agent.run_sync(prompt, timeout=300)
            custom_logger.debug(f"Agent output: {result.data}")
            return result.data
        finally:
            os.environ.pop('EXPECTED_FILE_PATH', None)

    @task
    def parse_agent_json_output(json_string: str) -> Dict[str, Any]:
        custom_logger.info(f"Received from agent (first 500 chars): {json_string[:500]}")
        try:
            cleaned_json_string = re.sub(r'```json\n|```', '', json_string).strip()
            result = json.loads(cleaned_json_string)
            custom_logger.info(f"Parsed agent JSON output")
            if result.get('metrics', {}).get('error_message'):
                custom_logger.warning(f"Agent error: {result['metrics']['error_message']}")
            return result
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}. Received: {json_string[:500]}..."
            custom_logger.error(error_msg)
            return {
                "content": "Error: Agent returned malformed JSON.",
                "metrics": {"error_message": error_msg}
            }

    @task
    def create_embeddings(parsed_result: Dict[str, Any]) -> list[float]:
        content = parsed_result.get('content', '')
        if not content or parsed_result.get('metrics', {}).get('error_message'):
            custom_logger.warning(f"Skipping embedding due to no content or error")
            return []
        try:
            custom_logger.info(f"Generating embedding for content (first 100 chars): '{content[:100]}...'")
            embeddings_client = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_vector = embeddings_client.embed_query(content)
            custom_logger.info(f"Generated embedding vector of dimension {len(embedding_vector)}")
            return embedding_vector
        except Exception as e:
            custom_logger.error(f"Embedding error: {e}", exc_info=True)
            return []

    @task
    def store_embeddings_in_faiss(parsed_result: Dict[str, Any], embedding_vector: list[float]):
        content = parsed_result.get('content', '')
        if not embedding_vector or not content:
            custom_logger.warning("Skipping FAISS storage due to empty content or embedding")
            return
        try:
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_dim = len(embedding_vector)
            os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
            try:
                vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
                custom_logger.info("Loaded existing FAISS index")
            except Exception:
                custom_logger.info("Creating new FAISS index")
                index = faiss.IndexFlatL2(embedding_dim)
                vector_store = FAISS(
                    embedding_function=embeddings_model,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
            vector_store.add_embeddings(text_embeddings=[(content, embedding_vector)])
            custom_logger.info(f"Added 1 entry to FAISS index. Total entries: {vector_store.index.ntotal}")
            vector_store.save_local(FAISS_INDEX_PATH)
            custom_logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        except Exception as e:
            custom_logger.error(f"FAISS storage error: {e}", exc_info=True)
            raise

    context_data = prepare_context()
    agent_json_result = process_document_with_agent(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    embedding_result = create_embeddings(parsed_agent_result)
    store_embeddings_in_faiss(parsed_result=parsed_agent_result, embedding_vector=embedding_result)

transcript_processor_dag = transcript_pipeline()
