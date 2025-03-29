from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
import os  # os module for environment variable handling
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()


MODEL_NAMES = [
    "qwen2.5:14b",
    "qwen2.5:7b"
]

tool_tavily = TavilySearchResults(max_results=5)

tools = [tool_tavily]

app = FastAPI(title="LangGraph API", version="0.1.0")


class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]

# endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically slects the model specified in the request.
    """

    if request.model_name not in MODEL_NAMES:
        return {"error": f"Invalid model name: {request.model_name}. Please select a valid model."}


    llm = ChatOllama(base_url="http://localhost:11434/", model=request.model_name)

    # il parametro state_modifier 
    ########################### aggiungi memoria con checkpointer=MemorySaver() e config per i threads
    agent = create_react_agent(model=llm, tools=tools, prompt=request.system_prompt)

    # create the initial state for processing 
    state = {
        "messages": request.messages,
    }

    result = agent.invoke(state)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
