@task
def classify(input_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Running classifier for: {input_path}")
    result = document_classifier_agent.run_sync(input_path)

    output = result.output.strip()
    logger.info(f"Raw agent output: {output}")

    # Clean triple backtick blocks (like ```json ... ```)
    if output.startswith("```"):
        output = output.strip("`")
        if output.lower().startswith("json"):
            output = output.split("\n", 1)[-1]

    try:
        data = json.loads(output)
        logger.info(f"Parsed output: {data}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
