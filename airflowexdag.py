from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dag_check',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2025, 5, 19),
    catchup=False,
) as dag:

    def print_message():
        print("DAG Test for PythonOperator")

    python_task = PythonOperator(
        task_id='Print_Messsage_task',
        python_callable=print_message,  # Fixed the typo here
    )    

    bash_task = BashOperator(
        task_id='bash_echo_task',
        bash_command='echo "DAG Test for BashOperator"',
    )

    python_task >> bash_task
