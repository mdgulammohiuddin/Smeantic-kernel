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
import logging
from typing import Dict, Any, Tuple

# --- Setup: NLTK, Logging, Environment Variables ---

# Download required NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Custom Logger Setup
CUSTOM_LOG_DIR_NAME = "dag_run_logs"
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_GUESS, "Backend")
CUSTOM_LOG_DIR = os.path.join(CUSTOM_LOG_BASE_DIR, CUSTOM_LOG_DIR_NAME)

# This is the central logger instance used throughout the DAG
custom_logger = logging.getLogger('TranscriptProcessorCustomLogger')
custom_logger.setLevel(logging.INFO)

def setup_custom_file_handler(logger_instance, log_file_name_prefix="transcript_processor"):
    """Sets up a file handler for the custom logger."""
    # This check prevents adding duplicate handlers on DAG reparses
    if not logger_instance.handlers:
        try:
            if not os.path.exists(CUSTOM_LOG_DIR):
                os.makedirs(CUSTOM_LOG_DIR, exist_ok=True)
                # No logger available yet, so use print for this specific, one-time message.
                print(f"Created custom log directory: {CUSTOM_LOG_DIR}")

            log_file_path = os.path.join(CUSTOM_LOG_DIR, f"{log_file_name_prefix}_{datetime.now().strftime('%Y%m%d')}.log")
            
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            
            logger_instance.info(f"--- Custom file logger initialized. Logging to: {log_file_path} ---")

        except Exception as e:
            # Use print as a fallback if logger setup itself fails
            print(f"CRITICAL: Error setting up custom file logger for '{logger_instance.name}': {e}")
            pass # Avoid crashing the application if logging fails

# Initialize the logger
setup_custom_file_handler(custom_logger)

# Configure verbosity of underlying libraries
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('unstructured').setLevel(logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    custom_logger.error("FATAL: OPENAI_API_KEY not set in environment variables.")
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
custom_logger.info("Successfully loaded OPENAI_API_KEY.")


ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
custom_logger.info(f"Assets directory configured at: {ASSETS_DIR}")

# --- Agent Tool Functions with Logging ---

def parse_document(file_path: str, expected_file_path: str = None) -> Tuple[str, Dict[str, Any]]:
    """Parse document and return content with metadata, validating the file path."""
    custom_logger.info(f"[Tool:parse_document] Starting execution for file: '{file_path}'")
    
    if expected_file_path and os.path.normpath(file_path) != os.path.normpath(expected_file_path):
        error_msg = f"Unauthorized file access attempted: '{file_path}'. Expected: '{expected_file_path}'"
        custom_logger.error(f"[Tool:parse_document] {error_msg}")
        return error_msg, {"error": error_msg}
        
    try:
        if not os.path.exists(file_path):
            error_msg = f"File not found at the specified path: {file_path}"
            custom_logger.error(f"[Tool:parse_document] {error_msg}")
            return error_msg, {"error": error_msg}
            
        elements = partition(filename=file_path)
        content = "\n\n".join([str(e) for e in elements])
        
        custom_logger.info(f"[Tool:parse_document] Successfully parsed document '{file_path}'. Raw content length: {len(content)} chars.")
        
        return content, {
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1].lstrip('.').upper()
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred during parsing for '{file_path}': {e}"
        custom_logger.error(f"[Tool:parse_document] {error_msg}", exc_info=True)
        return error_msg, {"error": error_msg}

def clean_content(raw_content: str) -> Tuple[str, Dict[str, Any]]:
    """Clean parsed content using NLTK for stopword and filler word removal."""
    custom_logger.info(f"[Tool:clean_content] Starting execution. Initial content length: {len(raw_content)} chars.")
    try:
        # 1. Remove timestamps
        timestamp_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?(\.\d{1,3})?\b'
        content = re.sub(timestamp_pattern, '', raw_content)
        
        # 2. Remove punctuation
        content = re.sub(r'[^\w\s]', '', content)
        
        # 3. Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 4. Remove stopwords and filler words
        stop_words = set(stopwords.words('english'))
        filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'i mean', 'kind of', 'sort of', 'well', 'okay', 'right'
        }
        words_to_remove = stop_words.union(filler_words)
        
        tokens = word_tokenize(content.lower())
        filtered_tokens = [word for word in tokens if word not in words_to_remove]
        cleaned = ' '.join(filtered_tokens)
        
        custom_logger.info(f"[Tool:clean_content] Content cleaning complete. Final length: {len(cleaned)} chars.")
        
        return cleaned, {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned)
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred during content cleaning: {e}"
        custom_logger.error(f"[Tool:clean_content] {error_msg}", exc_info=True)
        return error_msg, {"error": error_msg}

# --- Pydantic AI Agent Definition ---
custom_logger.info("Defining Pydantic AI Agent 'document_agent' with gpt-4o model.")
document_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[parse_document, clean_content],
    system_prompt="""
    You are a transcript processing system. Your goal is to process EXACTLY ONE document specified in the prompt, clean its content, and analyze it based on the user query.
    You MUST process ONLY the file path provided in the 'File Path' field of the prompt. Processing any other file is strictly forbidden and must result in an immediate error.
    You MUST return a single, valid JSON object (as a string) for the specified file only, with no Markdown formatting or code fences in the JSON structure.
    The JSON object must have two keys: "content" and "metrics".

    The "content" key must contain a string with query-relevant information formatted as Markdown bullet points.
    If an error occurs, "content" must contain a summary of the error.

    The "metrics" key must contain an object with:
      "file_type": (string, from parse_document metadata, or null if error)
      "file_size": (integer, from parse_document metadata, or null if error)
      "original_length": (integer, from clean_content metadata, or null if error before cleaning)
      "cleaned_length": (integer, from clean_content metadata, or null if error before cleaning)
      "error_message": (string, describe any error during parsing or cleaning, or null if no errors)

    Follow these steps meticulously:
    1. Extract the file path from the 'File Path' field in the prompt.
    2. Call `parse_document` with the validated file path, passing the same file path for both `file_path` and `expected_file_path` arguments to ensure security. If parsing fails, create and return an error JSON with appropriate metrics.
    3. If parsing succeeds, call `clean_content` on the parsed content. If cleaning fails, create and return an error JSON.
    4. If cleaning succeeds, analyze the cleaned content to answer the user query from the 'User Query' field. Generate the bullet-pointed analysis for the "content" key.
    5. Construct and return the final JSON string with the analysis and all collected metrics.

    Tool sequence: `parse_document` -> `clean_content` -> analysis.
    NEVER call `parse_document` for any file other than the one in the prompt's 'File Path' field.
    """
)
custom_logger.info("Pydantic AI Agent 'document_agent' has been defined.")

# --- Airflow DAG and Tasks with Logging ---

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

@dag(
    dag_id="transcript_processor_embed_fully_logged",
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
    """
    This DAG processes a document using an AI Agent, extracts structured data,
    generates embeddings, and stores them in a FAISS vector store.
    """
    custom_logger.info("DAG 'transcript_pipeline' is being parsed or executed.")

    @task
    def prepare_context(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves file path and prepares the initial context for the pipeline."""
        custom_logger.info("[Task:prepare_context] Starting...")
        params = kwargs.get('params', {})
        file_name = params.get('file_name')
        custom_logger.info(f"[Task:prepare_context] Received file_name parameter: '{file_name}'")

        if not file_name or (isinstance(file_name, str) and file_name.startswith("{{")):
            error_msg = f"Invalid or unrendered file_name parameter: '{file_name}'"
            custom_logger.error(f"[Task:prepare_context] {error_msg}")
            raise ValueError(error_msg)

        # Resolve file path
        candidate_path = os.path.join(ASSETS_DIR, file_name) if not os.path.isabs(file_name) else file_name
        custom_logger.info(f"[Task:prepare_context] Checking primary candidate path: '{candidate_path}'")
        
        if os.path.exists(candidate_path):
            file_path = candidate_path
            custom_logger.info(f"[Task:prepare_context] File found at primary path.")
        else:
            fallback_path = os.path.join(ASSETS_DIR, os.path.basename(file_name))
            custom_logger.warning(f"[Task:prepare_context] Primary path not found. Checking fallback: '{fallback_path}'")
            if os.path.exists(fallback_path):
                file_path = fallback_path
                custom_logger.info(f"[Task:prepare_context] File found at fallback path.")
            else:
                error_msg = f"File not found. Checked primary path '{candidate_path}' and fallback '{fallback_path}'"
                custom_logger.error(f"[Task:prepare_context] {error_msg}")
                raise FileNotFoundError(error_msg)

        user_query = params.get('user_query', "Summarize the document.")
        context = {
            "file_path": file_path,
            "user_query": user_query,
            "process_start_time": datetime.now(timezone.utc).isoformat()
        }
        custom_logger.info(f"[Task:prepare_context] Successfully prepared context: {context}")
        return context

    @task.agent(agent=document_agent)
    def process_document_with_agent(context: Dict[str, Any]) -> str:
        """This task formats and returns the prompt for the AI agent."""
        custom_logger.info("[Task:process_document_with_agent] Starting...")
        file_path = context['file_path']
        user_query = context['user_query']
        custom_logger.info(f"[Task:process_document_with_agent] Formatting prompt for agent with file: {file_path}")

        prompt = f"""
        User Query: {user_query}
        File Path: {file_path}
        """
        custom_logger.info("[Task:process_document_with_agent] Prompt created successfully. Returning to @task.agent operator for execution.")
        return prompt

    @task
    def parse_agent_json_output(json_string: str) -> Dict[str, Any]:
        """Parses the JSON string output from the agent into a dictionary."""
        custom_logger.info("[Task:parse_agent_json_output] Starting...")
        custom_logger.debug(f"[Task:parse_agent_json_output] Received raw string from agent: '{json_string}'")
        try:
            cleaned_json_string = re.sub(r'```json\n|```', '', json_string).strip()
            if cleaned_json_string != json_string:
                custom_logger.info("[Task:parse_agent_json_output] Cleaned markdown fences from JSON string.")
            
            result = json.loads(cleaned_json_string)
            custom_logger.info("[Task:parse_agent_json_output] Successfully parsed agent JSON output.")
            
            if result.get('metrics', {}).get('error_message'):
                custom_logger.warning(f"[Task:parse_agent_json_output] Agent reported a processing error: {result['metrics']['error_message']}")
            return result
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON from agent. Error: {e}. Received (first 500 chars): {json_string[:500]}..."
            custom_logger.error(f"[Task:parse_agent_json_output] {error_msg}")
            return {
                "content": "Fatal Error: Agent returned malformed JSON.",
                "metrics": {"error_message": error_msg}
            }

    @task
    def create_embeddings(parsed_result: Dict[str, Any]) -> list[float]:
        """Generate embeddings for the content if it's valid."""
        custom_logger.info("[Task:create_embeddings] Starting...")
        content = parsed_result.get('content', '')
        if not content or parsed_result.get('metrics', {}).get('error_message'):
            custom_logger.warning("[Task:create_embeddings] Skipping embedding due to no content or an upstream error.")
            return []
            
        try:
            custom_logger.info(f"[Task:create_embeddings] Generating embedding for content (first 100 chars): '{content[:100]}...'")
            embeddings_client = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_vector = embeddings_client.embed_query(content)
            custom_logger.info(f"[Task:create_embeddings] Successfully generated embedding vector of dimension {len(embedding_vector)}")
            return embedding_vector
        except Exception as e:
            custom_logger.error(f"[Task:create_embeddings] Failed during embedding generation: {e}", exc_info=True)
            return [] # Return empty list on failure to prevent downstream crashes

    @task
    def store_embeddings_in_faiss(parsed_result: Dict[str, Any], embedding_vector: list[float]):
        """Stores the text and its pre-computed embedding in an in-memory FAISS index."""
        custom_logger.info("[Task:store_embeddings_in_faiss] Starting...")
        content = parsed_result.get('content', '')
        if not embedding_vector or not content:
            custom_logger.warning("[Task:store_embeddings_in_faiss] Skipping FAISS storage due to empty content or embedding vector.")
            return

        try:
            custom_logger.info("[Task:store_embeddings_in_faiss] Initializing FAISS components.")
            embeddings_model_for_faiss = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
            embedding_dim = len(embedding_vector)
            
            index = faiss.IndexFlatL2(embedding_dim)
            vector_store = FAISS(
                embedding_function=embeddings_model_for_faiss,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            
            custom_logger.info(f"[Task:store_embeddings_in_faiss] Storing text and embedding vector (dim: {embedding_dim}) in FAISS index.")
            vector_store.add_embeddings(text_embeddings=[(content, embedding_vector)])
            custom_logger.info(f"[Task:store_embeddings_in_faiss] Successfully added entry to FAISS. Index now contains {vector_store.index.ntotal} entries.")
        except Exception as e:
            custom_logger.error(f"[Task:store_embeddings_in_faiss] An error occurred during FAISS storage: {e}", exc_info=True)
            raise

    # --- DAG Execution Flow ---
    custom_logger.info("Defining DAG task dependencies.")
    context_data = prepare_context()
    agent_json_result = process_document_with_agent(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    embedding_result = create_embeddings(parsed_agent_result)
    
    store_embeddings_in_faiss(
        parsed_result=parsed_agent_result, 
        embedding_vector=embedding_result
    )
    custom_logger.info("DAG task dependencies defined successfully.")

# Instantiate the DAG
transcript_processor_dag_logged = transcript_pipeline()
