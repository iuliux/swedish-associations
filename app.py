from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import answer_question  # Import function from retrieval.py

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    return {"answer": answer_question(req.question)}

