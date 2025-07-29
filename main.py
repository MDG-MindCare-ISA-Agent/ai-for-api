from fastapi import FastAPI
from pydantic import BaseModel
from emotion_service import clova_financial_chatbot

app = FastAPI()

class Input(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: Input):
    return clova_financial_chatbot(input.text)