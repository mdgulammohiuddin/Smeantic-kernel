import os
import json
import logging
import pickle
from airflow.decorators import dag, task
from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAIAgent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

# --- Setup: Logging, Environment Variables ---
CUSTOM_LOG_DIR_NAME = "dag_run_logs"
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_GUESS, "Backend")
CUSTOM_LOG_DIR = os.path.join(CUSTOM_LOG_BASE_DIR, CUSTOM_LOG_DIR_NAME)

custom_logger = logging.getLogger('FileQueryCustomLogger')
custom_logger.setLevel(logging.INFO)

def setup_custom_file_handler(logger_instance, log_file_name_prefix="file_query"):
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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    custom_logger.error("OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# FAISS storage paths (matching file_parsing_agent.py)
FAISS_DATA_DIR = os.path.join(PROJECT_ROOT_GUESS, "faiss_data")
FAISS_INDEX_FILE = os.path.join(FAISS_DATA_DIR, "persisted_faiss.index")
DOCSTORE_FILE = os.path.join(FAISS_DATA_DIR, "persisted_docstore.pkl")
INDEX_TO_ID_FILE = os.path.join(FAISS_DATA_DIR, "persisted_index_to_id.pkl")

# --- Tool Functions for the Agent ---
def generate_query_embedding(query: str) -> List[float]:
    """
    Generates an embedding for the input query using OpenAI's text-embedding-ada-002 model.
    
    Args:
        query (str): The user query to embed.
    
    Returns:
        List[float]: The embedding vector, or empty list if an error occurs.
    """
    custom_logger.info(f"Generating embedding for query: {query[:100]}...")
    if not query:
        custom_logger.warning("Empty query provided. Returning empty embedding.")
        return []
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        embedding_vector = embeddings_model.embed_query(query)
        custom_logger.info(f"Generated embedding with dimension {len(embedding_vector)}")
        return embedding_vector
    except Exception as e:
        custom_logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return []

def perform_faiss_search(query_embedding: List[float]) -> List[str]:
    """
    Loads the FAISS index and performs a similarity search using the query embedding.
    
    Args:
        query_embedding (List[float]): The query embedding vector.
    
    Returns:
        List[str]: List of retrieved document contents, or empty list if an error occurs.
    """
    custom_logger.info("Performing FAISS similarity search")
    if not query_embedding:
        custom_logger.warning("Empty query embedding provided. Returning empty results.")
        return []
    
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        # Check if all FAISS files exist
        if not all(os.path.exists(f) for f in [FAISS_INDEX_FILE, DOCSTORE_FILE, INDEX_TO_ID_FILE]):
            error_msg = f"FAISS data files missing in {FAISS_DATA_DIR}"
            custom_logger.error(error_msg)
            return []

        # Load FAISS components
        custom_logger.info(f"Loading FAISS index from {FAISS_INDEX_FILE}")
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(DOCSTORE_FILE, "rb") as f:
            docstore = pickle.load(f)
        with open(INDEX_TO_ID_FILE, "rb") as f:
            index_to_docstore_id = pickle.load(f)

        # Validate index dimension
        embedding_dim = 1536  # For text-embedding-ada-002
        if index.d != embedding_dim:
            error_msg = f"Index dimension {index.d} does not match expected {embedding_dim}"
            custom_logger.error(error_msg)
            return []

        # Initialize FAISS vector store
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        custom_logger.info(f"Loaded FAISS index with {vector_store.index.ntotal} entries")

        # Perform similarity search
        results = vector_store.similarity_search_by_vector(query_embedding, k=5)
        retrieved_docs = [doc.page_content for doc in results]
        custom_logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
    except Exception as e:
        custom_logger.error(f"FAISS search error: {e}", exc_info=True)
        return []

# --- Pydantic AI Agent with Tools ---
query_agent = PydanticAIAgent(
    model="gpt-4o",
    tools=[generate_query_embedding, perform_faiss_search],
    system_prompt="""
You are a query processing system for document retrieval from a FAISS index. Your goal is to retrieve relevant documents based on a user query and format the results into a structured JSON response. You MUST use the provided tools to perform the retrieval and MUST NOT process any files directly.

Input format:
User Query: [the user's query]

JSON output structure:
- "results": String with Markdown bullet points summarizing query-relevant content from retrieved documents. If empty, state "No relevant information found for the query."
- "metrics": Object with:
  - "num_retrieved": Integer, number of documents retrieved (0 if none).
  - "error_message": String, error description, or null if no error.

Steps:
1. Extract the user query from the input.
2. Call the `generate_query_embedding` tool with the user query to obtain the query embedding.
3. If the embedding is empty, return a JSON with "results": "No relevant information found due to embedding failure." and appropriate metrics.
4. Call the `perform_faiss_search` tool with the query embedding to retrieve documents.
5. If no documents are retrieved, return a JSON with "results": "No relevant information found for the query." and appropriate metrics.
6. Analyze the retrieved documents to answer the user query, summarizing relevant information in concise Markdown bullet points.
7. Construct and return a JSON string with the formatted results and metrics.

Example output:
{
  "results": "- Main subject: Project timeline updates.\n- Action item: Schedule meeting.",
  "metrics": {
    "num_retrieved": 2,
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
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=5),
}

@dag(
    dag_id="file_query_dag_with_tools",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["document_query", "pydantic-ai", "faiss", "query", "agent_tools"],
    params={
        "user_query": "What is the main subject of the document?"
    }
)
def file_query_pipeline():
    @task
    def prepare_context(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = kwargs.get('params', {})
        user_query = params.get('user_query', '').strip()
        if not user_query:
            error_msg = "User query is empty or missing"
            custom_logger.error(error_msg)
            raise ValueError(error_msg)
        custom_logger.info(f"Prepared context: user_query={user_query}")
        return {
            "user_query": user_query,
            "process_start_time": datetime.now(timezone.utc).isoformat()
        }

    @task.agent(agent=query_agent)
    def process_query_with_tools(context: Dict[str, str]) -> str:
        user_query = context['user_query']
        custom_logger.info(f"Processing query with agent tools: {user_query}")
        prompt = f"""
User Query: {user_query}
"""
        custom_logger.debug(f"Agent prompt: {prompt[:500]}...")
        return prompt

    @task
    def parse_agent_json_output(json_string: str) -> Dict[str, Any]:
        custom_logger.info(f"Agent output (first 500 chars): {json_string[:500]}")
        try:
            cleaned_json_string = json_string.replace('```json\n', '').replace('```', '').strip()
            result = json.loads(cleaned_json_string)
            custom_logger.info("Parsed agent JSON output")
            if result.get('metrics', {}).get('error_message'):
                custom_logger.warning(f"Agent error: {result['metrics']['error_message']}")
            return result
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}. Received: {json_string[:500]}..."
            custom_logger.error(error_msg)
            return {
                "results": "Error: Malformed JSON from agent",
                "metrics": {"num_retrieved": 0, "error_message": error_msg}
            }

    @task
    def format_final_output(result_dict: Dict[str, Any], context: Dict[str, str]) -> str:
        query_results = result_dict.get('results', 'No relevant information found.')
        metrics = result_dict.get('metrics', {})
        num_retrieved = metrics.get('num_retrieved', 0)
        error_message = metrics.get('error_message', None)
        output = f"""
## Document Query Report

### Query Metadata
- Query: {context['user_query']}
- Process Started: {context['process_start_time']}

### Query Metrics
- Documents Retrieved: {num_retrieved}
{f"- Error: {error_message}" if error_message else "No errors reported."}

### Query Results
{query_results}

### Final Output
Generated at: {datetime.now(timezone.utc).isoformat()}
"""
        print(output)
        custom_logger.info(f"Final report: {output[:200]}...")
        return output

    # DAG execution flow
    context_data = prepare_context()
    agent_json_result = process_query_with_tools(context_data)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    final_report = format_final_output(parsed_agent_result, context_data)

# Instantiate DAG
file_query_dag = file_query_pipeline()
