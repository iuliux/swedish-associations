import re
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Initialize Ollama LLM
llm = OllamaLLM(model="associations-rag")#, temperature=0.3)  # Using your custom trained model

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
Du är en AI som svarar på frågor på svenska med hjälp av juridiska dokument från bostadsrättsföreningar.
Om du inte vet svaret, säg att du inte vet.
Svar måste vara på svenska. Använd följande information för att besvara frågan:

{context}

Fråga: Strikt baserat på dokumenten, {question}
Kort svar:
    """
)

# Load Swedish-optimized model
model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

def highlight_relevant_sentences(question: str, text: str, top_k: int = 2) -> str:
    """Highlight sentences most semantically relevant to the question"""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if len(s) > 10]
    
    if not sentences:
        return text
        
    # Encode question and sentences
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(
        [question_embedding],
        sentence_embeddings
    )[0]
    
    # Find most relevant sentences
    top_indices = np.argsort(similarities)[-top_k:]
    highlighted = []
    
    for i, sentence in enumerate(sentences):
        if i in top_indices:
            highlighted.append(f"<strong>{sentence}</strong>")
        else:
            highlighted.append(sentence)
    
    return ' '.join(highlighted)


MIN_RELEVANCE_SCORE = 0.7  # Minimum relevance score for a chunk to be considered relevant

# Function to answer questions and retrieve source information
def answer_question(question: str, association: int):
    # Retrieve relevant documents based on the association
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {"association": {"$in": ["general", str(association)]}},
            # "k": 3,  # Number of documents to retrieve
        }
    )

    # Define the retrieval and generation chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Retrieve the most relevant chunks
    chunks_with_scores = retriever.get_relevant_documents_with_score(question, k=5)
    relevant_chunks = [
        chunk for chunk, score in chunks_with_scores 
        if score >= MIN_RELEVANCE_SCORE
    ]
    answer = question | rag_chain

    # Prepare source information
    sources = []
    for chunk in relevant_chunks:
        highlighted = highlight_relevant_sentences(question, chunk.page_content)
        sources.append({
            "text": highlighted,  # The text of the chunk
            "source": chunk.metadata["source"],  # The source document
            "page": chunk.metadata.get("page", None),  # Add page number if available
        })

    return {
        "answer": answer,
        "sources": sources
    }
