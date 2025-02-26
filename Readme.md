pip install fastapi uvicorn langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers ollama

cd workspace
git clone https://github.com/iuliux/swedish-associations.git .

apt update && apt install -y pciutils lshw
curl -fsSL https://ollama.ai/install.sh | sh

OLLAMA_ACCELERATE=1 ollama serve &

ollama create associations-rag
ollama show associations-rag

python server.py &
