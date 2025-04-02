from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


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

# Initialize a QA model to find relevant spans
qa_model = pipeline("question-answering", model="distilbert-base-multilingual-cased")

def highlight_relevant_text(question: str, text: str) -> str:
    """Adds bold markers around the most relevant part of text for the question"""
    try:
        result = qa_model(question=question, context=text)
        start, end = result["start"], result["end"]
        return text[:start] + "<strong>" + text[start:end] + "</strong>" + text[end:]
    except:
        return text  # Fallback if highlighting fails

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
        highlighted_text = highlight_relevant_text(question, chunk.page_content)
        sources.append({
            "text": highlighted_text,  # The text of the chunk
            "source": chunk.metadata["source"],  # The source document
            "page": chunk.metadata.get("page", None),  # Add page number if available
        })

    return {
        "answer": answer,
        "sources": sources
    }
