from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import answer_question  # Import function from retrieval.py

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        answer = answer_question(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
