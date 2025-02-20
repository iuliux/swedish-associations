import os
import sys
import json
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local("faiss_index", embeddings)

vectorstore = load_vectorstore()

# Query FAISS and Llama
def process_request():
    # Read input from Ollama
    raw_input = sys.stdin.read()
    request_data = json.loads(raw_input)
    question = request_data["prompt"]

    # Retrieve relevant context
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Construct prompt
    prompt = f"Använd följande information för att besvara frågan på svenska:\n{context}\n\nFråga: {question}\nSvar:"

    # Call Ollama API (self-hosted)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "associations-rag", "prompt": prompt}
    )

    output = response.json().get("response", "Ingen respons.")
    print(json.dumps({"response": output}))  # Ollama expects JSON output

if __name__ == "__main__":
    process_request()
