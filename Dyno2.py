@task.agent(agent=document_classifier_agent)
def classify(input_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Running classifier for: {input_path}")
    result = document_classifier_agent.run_sync(input_path)

    import json
    data = json.loads(result.data) if isinstance(result.data, str) else result.data

    logger.info(f"Agent output: {data}")
    return data
