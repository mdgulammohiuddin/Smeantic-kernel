from typing import Literal
from airflow.decorators import dag, task
import pendulum
from airflow.models.dagrun import DagRun


from dotenv import load_dotenv
import os
 
# Specify the absolute path to your .env file (e.g., in a parent or sibling folder)
dotenv_path = "/app/fdi/fdi_venv/.env"  # Absolute path
load_dotenv(dotenv_path)
 
# Access the API key
api_key = os.getenv("key")
model_name = os.getenv("name")
print(api_key)  # Verify it works
print(model_name)  


# Verify it works
@task.llm(
    model=model_name,
    result_type=Literal["positive", "negative", "neutral"],
    system_prompt="Classify the sentiment of the given text.",
)
def process_with_llm(dag_run: DagRun) -> str:
    input_text = dag_run.conf.get("input_text")

    # can do pre-processing here (e.g. PII redaction)
    return input_text

    
  
@dag(
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1),
    catchup=False,
    params={"input_text": "I'm very happy with the product."},
)
def sentiment_classification():
    process_with_llm()


sentiment_classification()
