from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths to your documents
documents_paths = [
    "documents/Bostadsrattslagen.txt",
    "documents/5/Stadgar.pdf",
    "documents/5/ordningsregler-2023.pdf",
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

    loaded_docs = loader.load()
    
    # Add custom metadata
    path_split = path.split("/")
    for doc in loaded_docs:
        doc.metadata["association"] = "general" if len(path_split) == 2 else path_split[1]
        doc.metadata["source"] = path_split[-1]
    
    documents.extend(loaded_docs)

# Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the index
vectorstore.save_local("faiss_index")

# Print details of vectorstore, like all documents, metadata, etc.
# Get all unique source documents
unique_sources = set()
unique_assoc = set()
for doc in vectorstore.docstore._dict.values():
    unique_sources.add(f'{doc.metadata["association"]} / {doc.metadata["source"]}')

print("Unique source documents in index:")
for source in sorted(unique_sources):
    print(f"- {source}")


print("FAISS index created and saved successfully.")