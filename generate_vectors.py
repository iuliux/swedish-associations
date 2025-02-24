from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Paths to your documents
documents_paths = [
    "documents/Bostadsrattslagen.txt",
    "documents/StadgarBRFvingarden.pdf",
    "documents/Ordningsregler.pdf",
]

# Load documents dynamically
documents = []
for path in documents_paths:
    if path.endswith(".txt"):
        loader = TextLoader(path)
    elif path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        print(f"Skipping unsupported file type: {path}")
        continue

    documents.extend(loader.load())

# Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the index
vectorstore.save_local("faiss_index")

print("FAISS index created and saved successfully.")