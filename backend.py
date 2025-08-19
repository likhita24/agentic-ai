# 1: Set up the schema validation using Pydantic
from pydantic import BaseModel
from typing import List
from ai_agent import call_agent

class AgentRequestSchema(BaseModel):
    model_name: str
    model_provider: str
    prompt: str
    messages: List[str]
    allow_search: bool

# 2: Set up AI Agent to handle request from the frontend
from fastapi import FastAPI

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app = FastAPI(title="AI Agent")

@app.post("/chat-agent")
def chat_agent_endpoint(request: AgentRequestSchema):
    """
    Endpoint to handle chat requests to the AI agent.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Model not supported. Please select a valid model."}

    # Call the agent with the provided parameters
    response = call_agent(
        llm_model=request.model_name,
        query=request.messages,
        allow_search=request.allow_search,
        system_prompt=request.prompt,
        provider=request.model_provider
    )

    return {response}

# 3: Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)