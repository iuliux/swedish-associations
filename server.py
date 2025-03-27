from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from retrieval import answer_question  # Import function from retrieval.py

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://helpful-frank.netlify.app", "https://03943d79-f799-4b33-a5f5-3c732a7ff38e.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class QuestionRequest(BaseModel):
    question: str
    association_id: int

@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        result = answer_question(req.question, req.association_id)
        return {
            "question": req.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
