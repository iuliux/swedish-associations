import re
import numpy as np
import traceback
from logger import logger

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import List, Dict, Any


# Load FAISS index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

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


MIN_CHUNK_SCORE = 0.4  # Minimum relevance score for a chunk to be considered relevant
MIN_SENTENCE_SCORE = 0.45  # Minimum relevance score for a chunk to be considered relevant

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
        # Logging the similarity score for each sentence
        logger.debug(f"{'TOP ' if i in top_indices else ''}Sentence: {sentence}\nSimilarity: {similarities[i]}\n\n")
        # if i in top_indices:
        if similarities[i] >= MIN_SENTENCE_SCORE:
            highlighted.append(f"<strong>{sentence}</strong>")
        else:
            highlighted.append(sentence)
    
    return ' '.join(highlighted)



@chain
def filtered_scored_retriever(input: Dict[str, Any]) -> List[Document]:
    """FAISS retriever with pre-filtering and scoring"""
    # Handle both direct params and LangChain input dict
    question = input.get("question") if isinstance(input, dict) else input
    association = input.get("association", "general") if isinstance(input, dict) else "general"
    k = input.get("k", 4) if isinstance(input, dict) else 4
    min_score = input.get("min_score", 0.5) if isinstance(input, dict) else 0.5

    # Logging the input parameters
    logger.debug(f"Query: {question}, Association: {association}, k: {k}, min_score: {min_score}")

    docs_with_scores = []
    query_embedding = embeddings.embed_query(question)
    
    for i, doc in enumerate(vectorstore.docstore._dict.values()):
        # Association filter comes first
        if doc.metadata.get("association") in ["general", association]:
            # Log the document ID and its metadata
            doc_embedding = vectorstore.index.reconstruct(i)
            score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            # logger.debug(f"Document ID: {i}, Score: {score}\n{doc.page_content[:500]}\n---\n")
            
            if score >= min_score:
                # Create new document to avoid modifying cached version
                scored_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": float(score)}
                )
                docs_with_scores.append((scored_doc, score))
    
    # Return top k by score
    return [doc for doc, _ in sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:k]]

# Function to answer questions and retrieve source information
def answer_question(question: str, association: int):
    try:
        # Wrap the retriever call with validation
        if not isinstance(question, str):
            raise ValueError(f"Question must be string, got {type(question)}")

        # Retrieve relevant documents based on the association
        retriever = filtered_scored_retriever

        # Define the retrieval and generation chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "association": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Invalid question: must be a non-empty string")

        # Retrieve the most relevant chunks
        relevant_chunks = retriever.invoke({
            "context": retriever,
            "question": question.strip(),
            "association": str(association),
            "k": 4,
            "min_score": MIN_CHUNK_SCORE
        })

        # Convert retrieved documents into a single text block
        context_text = "\n\n".join(doc.page_content for doc in relevant_chunks)
        # Invoke the chain with the question and association
        answer = rag_chain.invoke({
            "context": context_text,
            "question": question,
            "association": str(association)
        })

        # Prepare source information
        sources = []
        for chunk in relevant_chunks:
            highlighted = highlight_relevant_sentences(question, chunk.page_content)
            sources.append({
                "text": highlighted,  # The text of the chunk
                "source": chunk.metadata["source"],  # The source document
                "page": chunk.metadata.get("page", None),  # Add page number if available
                "score": chunk.metadata["score"],  # The relevance score
            })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}\n{traceback.format_exc()}")
        raise  # Re-raise for the HTTP handler
