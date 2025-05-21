import ast

@task.agent(agent=document_classifier_agent)
def classify(input_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Running classifier for: {input_path}")
    result = document_classifier_agent.run_sync(input_path)
    output_data = result.data
    
    if isinstance(output_data, str):
        try:
            data = json.loads(output_data)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(output_data)
            except Exception:
                logger.error(f"Failed to parse agent output: {output_data}")
                raise
    else:
        data = output_data

    logger.info(f"Agent output: {data}")
    return data
