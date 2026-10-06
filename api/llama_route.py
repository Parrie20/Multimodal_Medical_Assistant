from fastapi import APIRouter
from pydantic import BaseModel
import requests


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/api/v1/llm/chat")
def chat(request: ChatRequest):

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            "stream": False
        }
    )

    result = response.json()

    return {
        "response": result["message"]["content"]
    }