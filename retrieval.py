import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Initialize Ollama LLM
llm = Ollama(model="associations-rag")  # Using your custom trained model

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

def highlight_with_mistral(question: str, text: str, llm: Ollama) -> str:
    """Use Mistral to identify and mark the most relevant parts of the text"""
    # Split into manageable chunks (sentences or paragraphs)
    text_chunks = [chunk for chunk in re.split(r'(?<=[.!?])\s+', text) if len(chunk) > 15]
    
    if not text_chunks:
        return text
    
    # Prompt template for relevance detection
    highlight_prompt = ChatPromptTemplate.from_template("""
    [INST]Analysera följande textstycke i förhållande till frågan. 
    Markera bara den mest relevanta meningen med <strong>text</strong> taggar om den är direkt relaterad till frågan.
    
    Fråga: {question}
    Text: {chunk}
    
    Returnera endast den markerade texten eller originalet om inget är relevant.[/INST]
    """)
    
    highlighted = []
    for chunk in text_chunks:
        # Get Mistral's judgment
        chain = highlight_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "chunk": chunk})
        
        # Fallback to original if the model didn't highlight
        highlighted.append(result if '**' in result else chunk)
    
    return ' '.join(highlighted)

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
    relevant_chunks = retriever.get_relevant_documents(question)
    answer = rag_chain.invoke(question)

    # Prepare source information
    sources = []
    for chunk in relevant_chunks:
        highlighted = highlight_with_mistral(question, chunk.page_content, llm)
        sources.append({
            "text": highlighted,  # The text of the chunk
            "source": chunk.metadata["source"],  # The source document
            "page": chunk.metadata.get("page", None),  # Add page number if available
        })

    return {
        "answer": answer,
        "sources": sources
    }
