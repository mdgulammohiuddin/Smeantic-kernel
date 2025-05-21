from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import Pydantic AI and Unstructured
from pydantic_ai import Agent as PydanticAIAgent
from unstructured.partition.auto import partition

# --- Add standard Python logging for more verbosity ---
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG) # Specifically for pydantic-ai
logging.getLogger('openai').setLevel(logging.DEBUG) # If using OpenAI, its logs can be helpful
# --- End of logging setup ---

# Load environment variables
load_dotenv()

# Ensure OPENAI_API_KEY is explicitly set in the environment for Pydantic AI with OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    print("Warning: OPENAI_API_KEY not found in environment variables. Pydantic AI agent with OpenAI may not work.")

# Get the absolute path to the assets directory
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

# Tool function for parsing documents with Unstructured.io
def parse_document_with_unstructured(file_url: str) -> str:
    """
    Parses a document from a given file URL and returns the parsed data in markdown format using unstructured.io.

    Args:
        file_url (str): The URL or local path to the document.

    Returns:
        str: Parsed document data in markdown format, or an error message.
    """
    try:
        actual_file_path = file_url
        # If the file_url is a relative path, try to resolve it against ASSETS_DIR or CWD
        if not os.path.isabs(file_url) and not file_url.startswith(("http://", "https://")):
            path_in_assets_relative = os.path.join(ASSETS_DIR, file_url)
            path_in_assets_basename = os.path.join(ASSETS_DIR, os.path.basename(file_url))

            if os.path.exists(path_in_assets_relative):
                actual_file_path = path_in_assets_relative
            elif os.path.exists(path_in_assets_basename):
                actual_file_path = path_in_assets_basename
            elif os.path.exists(file_url):  # Check if it's a relative path from current working directory
                actual_file_path = os.path.abspath(file_url)
            else:
                return f"Error: File not found. Checked paths related to: {file_url}"
        
        if not os.path.exists(actual_file_path) and not actual_file_path.startswith(("http://", "https://")):
             return f"Error: File not found at resolved path: {actual_file_path}"

        print(f"Attempting to parse file: {actual_file_path}")
        elements = partition(filename=actual_file_path)
        markdown_content = "\\n\\n".join([str(element) for element in elements])
        return markdown_content
    except Exception as e:
        return f"An error occurred during parsing '{file_url}': {e}"

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
        "The user's query and file path will be provided in the input message as described above."
    )
)

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
    tags=['document_parsing', 'pydantic-ai', 'unstructured', 'airsdk_agent', 'query_based_extraction'],
    params={'user_query': 'What is the main subject of this document?'}
)
def file_parsing_using_pydantic_ai_agent():
    """
    DAG that uses a Pydantic AI agent with Unstructured.io to parse a document
    and extract information relevant to a user query.
    The Airflow Agent SDK (@task.agent) integrates the Pydantic AI agent as a task.
    """

    @task.agent(agent=file_parser_pydantic_agent)
    def run_document_parsing_agent(file_path_to_parse: str, user_query: str) -> str:
        """
        Airflow task that runs the Pydantic AI agent.
        It formats the file_path_to_parse and user_query into a single string,
        which is then passed as the input message to the agent.
        """
        # The agent will receive this formatted string as its input query.
        # The system prompt guides it on how to parse this string and use the tool.
        return f"User Query: {user_query}\\nFile Path: {file_path_to_parse}"

    @task
    def show_parsed_document_content(parsed_content: str):
        """
        Task to print the parsed document content from the agent.
        """
        print("------ Parsed Document Output ------")
        print(parsed_content)
        print("------ End of Parsed Document Output ------")
        # You can also use XComs to pass this data or store it elsewhere.
        return parsed_content

    # Define the path to the document to be parsed
    # Ensure this file exists in your 'assets' directory relative to the DAGs folder
    # Example: /app/fdi/assets/safia_ajk_cc 1.pdf
    document_to_parse = os.path.join(ASSETS_DIR, "KB_0000268.pdf")
    
    # A preliminary check to see if the file exists, useful for debugging.
    # The tool itself also has error handling for file not found.
    if not os.path.exists(document_to_parse):
        print(f"Warning: The specified document '{document_to_parse}' does not seem to exist. The parsing task might fail.")

    # Run the agent task with the specified document path and user query
    parsed_output = run_document_parsing_agent(
        file_path_to_parse=document_to_parse,
        user_query="{{ params.user_query }}"
    )

    # Show the output
    show_parsed_document_content(parsed_content=parsed_output)

# Instantiate the DAG
file_parsing_dag_instance = file_parsing_using_pydantic_ai_agent()
