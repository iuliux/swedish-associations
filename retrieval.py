from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Initialize Ollama LLM
llm = Ollama(model="associations-rag")  # Using your custom trained model

# Create RetrievalQA pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" just inserts retrieved docs into the prompt
    retriever=vectorstore.as_retriever()
)

# Function to answer questions
def answer_question(question: str):
    return qa_chain.invoke(question)
