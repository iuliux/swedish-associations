1 x RTX A4000
8 vCPU 41 GB RAM
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
On-Demand - Community Cloud

12BG storage
Expose port 8000


cd workspace
git clone https://github.com/iuliux/swedish-associations.git .

pip install -r requirements.txt

apt update && apt install -y pciutils lshw
curl -fsSL https://ollama.ai/install.sh | sh

OLLAMA_ACCELERATE=1 ollama serve &

ollama create associations-rag
ollama show associations-rag

python server.py &
