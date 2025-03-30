from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
import os  # os module for environment variable handling
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


MODEL_NAMES = [
    "qwen2.5:14b",
    "qwen2.5:7b"
]

tool_tavily = TavilySearchResults(max_results=5)

tools = [tool_tavily]

chat_history = []

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

    # per collegare a ollama in running sull'host non in docker
    llm = ChatOllama(base_url="http://host.docker.internal:11434/", model=request.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", request.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt.format(chat_history=chat_history))

    # create the initial state for processing 
    state = {"messages": request.messages}

    result = agent.invoke(input=state)

    chat_history.append(AIMessage(content=result['messages'][-1].content))

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
