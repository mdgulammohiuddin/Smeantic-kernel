from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import pickle # Added for serializing docstore and index_to_id_map

# Import Pydantic AI and Unstructured
from pydantic_ai import Agent as PydanticAIAgent
from unstructured.partition.auto import partition

# --- Add new imports for FAISS and Langchain ---
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
# --- End of new imports ---

# --- Project-specific Custom Logger Setup ---
CUSTOM_LOG_DIR_NAME = "dag_run_logs"
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_GUESS, "Backend") 
CUSTOM_LOG_DIR = os.path.join(CUSTOM_LOG_BASE_DIR, CUSTOM_LOG_DIR_NAME)

# --- Define FAISS data storage paths ---
FAISS_DATA_DIR = os.path.join(PROJECT_ROOT_GUESS, "faiss_data")
FAISS_INDEX_FILE = os.path.join(FAISS_DATA_DIR, "persisted_faiss.index")
DOCSTORE_FILE = os.path.join(FAISS_DATA_DIR, "persisted_docstore.pkl")
INDEX_TO_ID_FILE = os.path.join(FAISS_DATA_DIR, "persisted_index_to_id.pkl")
# --- End of FAISS data storage paths ---

custom_logger = logging.getLogger('FileParserCustomLogger')
custom_logger.setLevel(logging.INFO)

def setup_custom_file_handler(logger_instance, log_file_name_prefix="file_parser"):
    if not logger_instance.handlers:
        try:
            if not os.path.exists(CUSTOM_LOG_DIR):
                os.makedirs(CUSTOM_LOG_DIR, exist_ok=True)
            
            log_file_path = os.path.join(CUSTOM_LOG_DIR, f"{log_file_name_prefix}_{datetime.now().strftime('%Y%m%d')}.log")
            
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(f"Custom file logger initialized for file parsing. Logging to: {log_file_path}")
        except Exception as e:
            # Use a basic print here if logger setup itself fails
            print(f"CRITICAL: Error setting up custom file logger for file_parsing_airflow_agent: {e}") 
            pass

setup_custom_file_handler(custom_logger)
# --- End of Custom Logger Setup ---

# --- Add standard Python logging for more verbosity if needed ---
# logging.basicConfig(level=logging.DEBUG) # Custom logger is now primary
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG) 
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('unstructured').setLevel(logging.INFO) # Keep unstructured logs manageable
# --- End of logging setup ---

# Load environment variables
load_dotenv()

# Ensure OPENAI_API_KEY is explicitly set in the environment for Pydantic AI with OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    custom_logger.warning("OPENAI_API_KEY not found in environment variables. Pydantic AI agent with OpenAI may not work.")

# Get the absolute path to the assets directory
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# Tool function for parsing documents with Unstructured.io
def parse_document_with_unstructured(file_url: str) -> str:
    """
    Parses a document from a given file URL (expected to be an absolute path)
    and returns the parsed data in markdown format using unstructured.io.

    Args:
        file_url (str): The absolute local path to the document.

    Returns:
        str: Parsed document data in markdown format, or an error message.
    """
    custom_logger.info(f"parse_document_with_unstructured: Received absolute file_url/path: '{file_url}'")

    if not file_url or not isinstance(file_url, str):
        error_msg = f"parse_document_with_unstructured: Invalid file_url received: '{file_url}'. Expected a valid string path."
        custom_logger.error(error_msg)
        return error_msg
    
    if not os.path.isabs(file_url):
        error_msg = f"parse_document_with_unstructured: Path is not absolute: '{file_url}'. This tool expects an absolute path."
        custom_logger.error(error_msg)
        return error_msg # Or raise an error

    if not os.path.exists(file_url):
        error_msg = f"parse_document_with_unstructured: Error - File not found at provided absolute path: {file_url}"
        custom_logger.error(error_msg)
        return error_msg # Or raise an error

    custom_logger.info(f"parse_document_with_unstructured: Attempting to parse resolved file: {file_url} with hi_res strategy.")
    try:
        elements = partition(filename=file_url, strategy="hi_res")
        markdown_content = "\n\n".join([str(element) for element in elements])
        custom_logger.info(f"parse_document_with_unstructured: Successfully parsed {len(elements)} elements from {file_url}. Preview (first 300 chars): {markdown_content[:300]}{'...' if len(markdown_content) > 300 else ''}")
        return markdown_content
    except Exception as e:
        error_msg = f"parse_document_with_unstructured: An error occurred during parsing '{file_url}': {e}"
        custom_logger.error(error_msg, exc_info=True)
        return error_msg

# Define the Pydantic AI agent for file parsing
# Ensure OPENAI_API_KEY is set in your environment for this to work
file_parser_pydantic_agent = PydanticAIAgent(
    model="gpt-4o",  # Using OpenAI GPT-4o model
    tools=[parse_document_with_unstructured],
    system_prompt=(
        "You are an expert document processing assistant. "
        "You will receive a combined input string, formatted as follows:\\n"
        "User Query: [the user's actual query]\\n"
        "File Path: [the actual file path]\\n\\n"
        "Your tasks are:\\n"
        "1. Extract the 'File Path' from this input.\\n"
        "2. Use the 'parse_document_with_unstructured' tool with this 'File Path' to get the full content of the document.\\n"
        "3. Extract the 'User Query' from the input.\\n"
        "4. After obtaining the full document content, analyze this content in conjunction with the 'User Query'.\\n"
        "5. Extract and return ONLY concise yet detailed information from the document that is directly relevant to the 'User Query'. Give in a structured format including all the important details and requisites are mentioned.\\n"
        "6. Do NOT add any conversational phrases like 'Here is the information...'. Simply return the extracted relevant data. If no relevant information is found, return a message like 'No relevant information found for the query.'\\n"
        "7. If the document contains email, then extract the to, from, subject and body and give whatever relevant information isn as per user query.\\n"
        "The user's query and file path will be provided in the input message as described above."
    )
)

# --- Globally Defined Airflow Tasks ---
@task.agent(agent=file_parser_pydantic_agent)
def run_document_parsing_agent(file_path_to_parse: str, user_query: str) -> str:
    """
    Airflow task (defined globally) that runs the Pydantic AI agent for document parsing.
    It formats the file_path_to_parse and user_query into a single string for the agent.
    """
    custom_logger.info(f"run_document_parsing_agent: Starting for query='{user_query}', file='{file_path_to_parse}'")
    return f"User Query: {user_query}\nFile Path: {file_path_to_parse}"

@task
def show_parsed_document_content(parsed_content: str):
    """
    Airflow task (defined globally) to log the parsed document content from the agent.
    """
    custom_logger.info("------ Parsed Document Output (Global Task Structure) ------")
    # Log potentially large content in a more controlled way, e.g. first 1000 chars
    # Or consider logging a summary/confirmation instead of full content if too verbose
    if parsed_content:
        custom_logger.info(f"{parsed_content[:1000]}{'...' if len(parsed_content) > 1000 else ''}")
    else:
        custom_logger.info("Parsed content is empty or None.")
    custom_logger.info("------ End of Parsed Document Output (Global Task Structure) ------")
    # This task is primarily for logging, so it doesn't need to return the content again
    # unless another task downstream needs it via XComs.

# --- New Task to Prepare Document Path ---
@task
def prepare_document_path_task(file_name_param: str) -> str:
    """
    Resolves the file name parameter to an absolute file path.
    Raises FileNotFoundError if the file cannot be located after checking various possibilities.
    """
    custom_logger.info(f"prepare_document_path_task: Received file_name_param: '{file_name_param}'")

    if not file_name_param or (isinstance(file_name_param, str) and file_name_param.startswith("{{")):
        error_msg = f"prepare_document_path_task: file_name_param is invalid or not properly templated: '{file_name_param}'."
        custom_logger.error(error_msg)
        raise ValueError(error_msg)

    candidate_path = ""
    if file_name_param.startswith("assets/"):
        candidate_path = os.path.abspath(os.path.join(PROJECT_ROOT_GUESS, file_name_param))
    elif os.path.isabs(file_name_param):
        candidate_path = file_name_param
    else:
        candidate_path = os.path.join(ASSETS_DIR, file_name_param)

    if os.path.exists(candidate_path):
        custom_logger.info(f"prepare_document_path_task: File found at: {candidate_path}")
        return candidate_path
    else:
        # Try fallback: ASSETS_DIR + basename
        fallback_path = os.path.join(ASSETS_DIR, os.path.basename(file_name_param))
        if os.path.exists(fallback_path):
            custom_logger.info(f"prepare_document_path_task: Main candidate '{candidate_path}' not found. Using fallback: '{fallback_path}'")
            return fallback_path
        else:
            error_msg = (f"prepare_document_path_task: File not found. "
                         f"Input param: '{file_name_param}', Checked primary: '{candidate_path}', Checked fallback: '{fallback_path}'.")
            custom_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
# --- End of New Task to Prepare Document Path ---

# --- New Globally Defined Airflow Tasks for Embedding and FAISS ---
@task
def create_embeddings_task(text_to_embed: str) -> list[float]:
    """
    This task generates embeddings for the given text using OpenAI's text-embedding-ada-002 model.
    It directly uses the OpenAIEmbeddings client.
    The XCom output of this task will be the embedding vector (list[float]).
    """
    custom_logger.info(f"create_embeddings_task: Starting to generate embedding for text (first 100 chars): '{text_to_embed[:100]}{'...' if len(text_to_embed) > 100 else ''}'")

    if not text_to_embed:
        custom_logger.warning("create_embeddings_task: Text to embed is empty. Returning empty list.")
        return [] # Return an empty list for empty input

    if not OPENAI_API_KEY:
        custom_logger.error("create_embeddings_task: OPENAI_API_KEY not found. Cannot initialize OpenAIEmbeddings.")
        raise ValueError("OPENAI_API_KEY is not configured, which is required for generating OpenAI embeddings.")

    try:
        embeddings_client = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        embedding_vector = embeddings_client.embed_query(text_to_embed)
        custom_logger.info(f"create_embeddings_task: Successfully generated embedding. Vector dimension: {len(embedding_vector)}")
        return embedding_vector
    except Exception as e:
        custom_logger.error(f"create_embeddings_task: Error during embedding generation: {e}", exc_info=True)
        raise

@task
def store_embeddings_in_faiss_task(original_text: str, embedding_vector: list[float]):
    """
    Stores the provided text and its pre-computed embedding in a persistent FAISS index.
    Loads existing index from local files if available, otherwise creates a new one.
    Saves the updated index, docstore, and ID mapping to local files.
    Accepts original_text and embedding_vector as separate arguments.
    """
    if not embedding_vector:
        custom_logger.warning("store_embeddings_in_faiss_task: Received empty embedding vector. Skipping FAISS storage.")
        return

    custom_logger.info(f"store_embeddings_in_faiss_task: Storing embedding in FAISS. Text (first 100 chars): '{original_text[:100]}{'...' if len(original_text) > 100 else ''}'")

    if not OPENAI_API_KEY:
        custom_logger.error("store_embeddings_in_faiss_task: OPENAI_API_KEY not found. Cannot initialize OpenAIEmbeddings.")
        raise ValueError("OPENAI_API_KEY is not configured for FAISS store setup.")

    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        # Known dimension for text-embedding-ada-002. Required for initializing a new index.
        embedding_dim = 1536 

        index: faiss.Index | None = None
        docstore: InMemoryDocstore | None = None
        index_to_docstore_id: dict[int, str] | None = None

        # Ensure the FAISS_DATA_DIR exists for loading; will be created for saving if it doesn't.
        os.makedirs(FAISS_DATA_DIR, exist_ok=True)

        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCSTORE_FILE) and os.path.exists(INDEX_TO_ID_FILE):
            custom_logger.info(f"store_embeddings_in_faiss_task: Loading existing FAISS index, docstore, and ID map from {FAISS_DATA_DIR}")
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(DOCSTORE_FILE, "rb") as f:
                docstore = pickle.load(f)
            with open(INDEX_TO_ID_FILE, "rb") as f:
                index_to_docstore_id = pickle.load(f)
            
            # Basic validation, e.g., check if index dimension matches expected if possible (FAISS API specific)
            if index is not None and index.d != embedding_dim:
                custom_logger.warning(f"Loaded FAISS index dimension {index.d} does not match expected embedding dimension {embedding_dim}. Re-initializing.")
                index = None # Force re-initialization
                docstore = None
                index_to_docstore_id = None

        else:
            custom_logger.info(f"store_embeddings_in_faiss_task: No complete existing FAISS data found in {FAISS_DATA_DIR}. Initializing new.")

        if index is None or docstore is None or index_to_docstore_id is None:
            custom_logger.info(f"store_embeddings_in_faiss_task: Initializing new FAISS index, docstore, and ID map with dimension {embedding_dim}.")
            index = faiss.IndexFlatL2(embedding_dim)
            docstore = InMemoryDocstore() # type: ignore
            index_to_docstore_id = {}    # type: dict[int, str]

        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        
        vector_store.add_embeddings(text_embeddings=[(original_text, embedding_vector)])

        custom_logger.info(f"store_embeddings_in_faiss_task: Successfully added text and embedding. Index now has {vector_store.index.ntotal} entries.")

        # Save the updated index, docstore, and ID mapping
        faiss.write_index(vector_store.index, FAISS_INDEX_FILE)
        with open(DOCSTORE_FILE, "wb") as f:
            pickle.dump(vector_store.docstore, f)
        with open(INDEX_TO_ID_FILE, "wb") as f:
            pickle.dump(vector_store.index_to_docstore_id, f)
        
        custom_logger.info(f"store_embeddings_in_faiss_task: Successfully saved FAISS index, docstore, and ID map to {FAISS_DATA_DIR}")
        
    except Exception as e:
        custom_logger.error(f"store_embeddings_in_faiss_task: Error during FAISS operations: {e}", exc_info=True)
        raise
# --- End of New Globally Defined Airflow Tasks ---

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    default_args=default_args,
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['document_parsing', 'pydantic-ai', 'unstructured', 'airsdk_agent', 'query_based_extraction', 'global_tasks'],
    params={
        'user_query': 'What is the main subject of this document?',
        'file_name': 'Copilot.docx' # Default file name parameter
    }
)
def file_parsing_agent():
    """
    DAG that uses a Pydantic AI agent with Unstructured.io to parse a document
    and extract information relevant to a user query.
    This DAG calls globally defined tasks.
    """
    custom_logger.info(f"DAG 'file_parsing_using_pydantic_ai_agent' starting.")

    # Resolve the file path using the new dedicated task.
    # {{ params.file_name }} will be templated by Airflow when this task runs.
    resolved_document_path = prepare_document_path_task(
        file_name_param="{{ params.file_name }}"
    )

    # The Pydantic AI agent will receive the fully resolved path.
    parsed_output = run_document_parsing_agent(
        file_path_to_parse=resolved_document_path,
        user_query="{{ params.user_query }}"
    )

    # Show the output
    show_parsed_document_content(parsed_content=parsed_output)

    # New tasks for embedding and FAISS storage
    # create_embeddings_task decorated with @task.embed will output the embedding vector directly.
    embedding_vector_output = create_embeddings_task(text_to_embed=parsed_output)
    store_embeddings_in_faiss_task(
        original_text=parsed_output, 
        embedding_vector=embedding_vector_output
    )

# Instantiate the DAG
dag = file_parsing_agent()
