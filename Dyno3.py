from typing import List, Dict
from pydantic import BaseModel
from pydantic_ai import tool
from airflow.decorators import dag, task
from airflow.models.dagrun import DagRun
import pendulum, logging

# Define a Pydantic model for classification results
class ClassificationResult(BaseModel):
    agent: str
    path: str

# Tool function to detect images in a document
@tool
def detect_images_in_document(path: str) -> bool:
    """
    Returns True if a local .pdf, .pptx, or .xlsx file contains images.
    (Here we simulate detection; URLs are assumed True by the agent logic.)
    """
    # If URL (starts with http/https), agent will assume True externally.
    if path.lower().startswith(("http://", "https://")):
        return False
    # For local files, rudimentary check based on extension (stub logic).
    if path.lower().endswith(('.pdf', '.pptx', '.xlsx')):
        # In a real implementation, inspect the file content for images.
        # Here we simply return False (no images) for demonstration.
        return False
    return False

# Create the Pydantic AI Agent with our classification logic
document_classifier_agent = PydanticAIAgent(
    model="gpt-4o",  # Use GPT-4o model for classification
    system_prompt=(
        "You are a document classification assistant. "
        "Classify the given path according to these rules:\n"
        "- Local files ending in .pdf, .docx, .ppt, .pptx, .xlsx get label 'File Parsing Agent'.\n"
        "- Additionally, .pdf, .pptx, .xlsx get 'Image Processing Agent' if images are detected (use detect_images_in_document tool). URLs always have images by assumption.\n"
        "- URLs (paths starting with http/https) always get label 'SharePoint Agent'.\n"
        "- All other or unknown extensions get label 'File Parsing Agent'.\n"
        "Return a JSON array of objects with fields 'agent' and 'path' for each assigned label."
    ),
    tools=[detect_images_in_document],  # Register the tool with the agent
    # Declare output type for automatic parsing (list of ClassificationResult)
    output_model=List[ClassificationResult],
)

@dag(
    dag_id="document_classifier",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
)
def doc_classifier_dag():
    # 1. Read input paths from DAG run configuration
    @task
    def get_input_paths(dag_run: DagRun) -> List[str]:
        # Expecting a list of file paths or URLs under 'paths' key
        return dag_run.conf.get("paths", [])

    # 2. Agent task: classify each document path
    @task.agent(document_classifier_agent)
    def classify_document(path: str) -> List[ClassificationResult]:
        # The agent logic is executed automatically; we just return the path as input.
        return path

    # 3. Flatten the list of lists of results
    @task
    def flatten_results(results: List[List[ClassificationResult]]) -> List[ClassificationResult]:
        # Flatten list of lists into a single list
        flattened = [item for sublist in results for item in sublist]
        return flattened

    # 4. Log each classification result
    @task
    def log_results(flattened: List[ClassificationResult]) -> List[ClassificationResult]:
        for res in flattened:
            logging.info(f"Assigned agent: {res.agent}, Path: {res.path}")
        return flattened

    # 5. Aggregate results by agent type
    @task
    def aggregate_results(flattened: List[ClassificationResult]) -> Dict[str, List[str]]:
        aggr: Dict[str, List[str]] = {}
        for res in flattened:
            aggr.setdefault(res.agent, []).append(res.path)
        return aggr

    # Define task dependencies / workflow
    paths = get_input_paths()
    classified = classify_document.expand(path=paths)
    flat = flatten_results(classified)
    logged = log_results(flat)
    aggregated = aggregate_results(logged)

    # We could return or push aggregated for downstream use
    return aggregated

doc_classifier_dag = doc_classifier_dag()
