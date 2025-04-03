from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from retrieval import answer_question  # Import function from retrieval.py
from logger import logger
import traceback


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://helpful-frank.netlify.app", "https://03943d79-f799-4b33-a5f5-3c732a7ff38e.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions"""
    error_detail = {
        "error_type": str(type(exc)),
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "request": {
            "url": str(request.url),
            "method": request.method,
            "body": await request.body() if request.method == "POST" else None
        }
    }
    
    logger.error(f"Unhandled exception: {error_detail}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "details": error_detail if request.query_params.get("debug") else None
        }
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
        # Log the full error with context
        error_context = {
            "input_question": req.question,
            "input_association": req.association,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "vectorstore_info": {
                "doc_count": len(vectorstore.docstore._dict) if hasattr(vectorstore, 'docstore') else None,
                "index_type": type(vectorstore.index).__name__ if hasattr(vectorstore, 'index') else None
            }
        }
        
        logger.error(f"RAG pipeline failed: {error_context}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "context": error_context if req.debug else None
            }
        )

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
