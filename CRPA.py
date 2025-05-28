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
CUSTOM_LOG_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "Backend")
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

# --- Pydantic AI Agent for Query-Specific Processing ---
query_agent = PydanticAIAgent(
    model="gpt-4o",
    system_prompt="""
You are an advanced query processing system for document retrieval from a FAISS index. Your goal is to analyze retrieved documents and generate a structured JSON response that directly addresses the user query with high relevance and precision. You MUST NOT process any files or invoke file-related tools. Your input consists of the user query and a list of retrieved document contents from FAISS.

JSON structure:
- "results": String with Markdown bullet points containing only information directly relevant to the user query. If no relevant information is found, state "No relevant information found for the query."
- "metrics": Object with:
  - "num_retrieved": Integer, number of documents retrieved (0 if none).
  - "error_message": String, error description, or null if no error.

Steps:
1. Extract the user query and retrieved documents from the input.
2. Interpret the query's intent (e.g., question type, specific document reference like "copilot document").
3. If no documents are retrieved, return a JSON with "results": "No relevant information found for the query." and "num_retrieved": 0.
4. Analyze each document to identify content that directly answers the query. Focus on:
   - Specific details requested (e.g., "main subject" should extract the primary topic).
   - References to named entities in the query (e.g., "copilot" should prioritize documents mentioning "Copilot").
   - Exclude irrelevant or tangentially related information.
5. Synthesize a concise response in Markdown bullet points, ensuring each point addresses the query directly.
6. If the query references a specific document (e.g., "copilot document"), prioritize content from documents matching that reference, if available.
7. Construct and return a JSON string with the formatted results and metrics.

Example input:
User Query: What is the main subject of the copilot document?
Retrieved Documents: ["Copilot enhances productivity with AI tools.", "Budget review for 2025."]

Example output:
{
  "results": "- Main subject of the copilot document: Enhancing productivity with AI tools.",
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
    dag_id="retreiving_file_query_dag",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["document_query", "pydantic-ai", "faiss", "query"],
    params={
        "user_query": "What is the main subject of the copilot document?"
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

    @task
    def retrieve_from_faiss(context: Dict[str, str]) -> List[str]:
        user_query = context['user_query']
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

            # Generate query embedding
            custom_logger.info(f"Generating embedding for query: {user_query}")
            query_embedding = embeddings_model.embed_query(user_query)

            # Perform similarity search
            custom_logger.info("Performing similarity search")
            results = vector_store.similarity_search_by_vector(query_embedding, k=5)
            retrieved_docs = [doc.page_content for doc in results]
            custom_logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
        except Exception as e:
            error_msg = f"FAISS retrieval error: {e}"
            custom_logger.error(error_msg, exc_info=True)
            return []

    @task.agent(agent=query_agent)
    def process_query_results(context: Dict[str, str], retrieved_docs: List[str]) -> str:
        user_query = context['user_query']
        custom_logger.info(f"Processing {len(retrieved_docs)} documents for query: {user_query}")
        prompt = f"""
**User Query**: {user_query}

**Retrieved Documents**:
{json.dumps(retrieved_docs, indent=2)}

**Instructions**:
- Analyze the retrieved documents to extract information that directly answers the user query.
- Focus on the query's intent and prioritize content relevant to any specific references (e.g., "copilot document").
- Return a JSON string as specified in the system prompt.
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
    retrieved_docs = retrieve_from_faiss(context_data)
    agent_json_result = process_query_results(context_data, retrieved_docs)
    parsed_agent_result = parse_agent_json_output(agent_json_result)
    final_report = format_final_output(parsed_agent_result, context_data)

# Instantiate DAG
file_query_dag = file_query_pipeline()
