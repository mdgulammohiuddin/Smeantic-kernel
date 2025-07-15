# main.py
import os
import time
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import FunctionTool
from dashboard import get_resource_name_by_partial_match

load_dotenv()

app = FastAPI()

PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

AGENT_ID_FILE = "agent_id.json"


class CreateAgentRequest(BaseModel):
    name: str
    instructions: str


class ChatRequest(BaseModel):
    query: str
    agent_id: str
    thread_id: str = ""  # Optional reuse


def get_client():
    credential = ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET
    )
    return AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)


@app.post("/create-agent")
async def create_agent(request: CreateAgentRequest):
    try:
        client = get_client()

        tool = FunctionTool(functions=[get_resource_name_by_partial_match])

        agent = client.agents.create_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name=request.name,
            instructions=request.instructions,
            tools=tool.definitions
        )

        # Save agent ID for future use
        with open(AGENT_ID_FILE, 'w') as f:
            json.dump({"agent_id": agent.id}, f)

        return {
            "agent_id": agent.id,
            "name": agent.name,
            "model": agent.model,
            "status": "Agent created successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        client = get_client()

        agent_id = request.agent_id
        thread_id = request.thread_id

        # If thread_id is not provided or fake, create a new one
        if not thread_id or thread_id == "thread123":
            thread = client.threads.create()
            thread_id = thread.id

        # Send user message
        message = client.messages.create(
            thread_id=thread_id,
            role="user",
            content=request.query
        )

        # Start run
        run = client.runs.create(
            thread_id=thread_id,
            assistant_id=agent_id
        )

        # Poll until run completes
        max_attempts = 20
        for _ in range(max_attempts):
            time.sleep(1)
            run = client.runs.get(thread_id=thread_id, run_id=run.id)

            if run.status == "completed":
                break
            elif run.status == "failed":
                raise HTTPException(status_code=500, detail="Run failed")

        # Fetch assistant reply
        messages = client.messages.list(thread_id=thread_id)
        for msg in messages:
            if msg.role == "assistant":
                content = msg.content[0].text.value if msg.content else ""
                return {
                    "response": content,
                    "thread_id": thread_id
                }

        return {
            "response": "No assistant response found.",
            "thread_id": thread_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/")
async def root():
    return {
        "message": "Azure Foundry Agent API is live",
        "endpoints": {
            "POST /create-agent": "Create a new agent",
            "POST /chat": "Chat with a created agent"
        }
    }
