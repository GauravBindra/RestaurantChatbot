import json
import os
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ------------------------- Load Wikipedia Knowledge Base -------------------------

SAVE_DIR = os.path.join("Data", "Wikipedia_data2")
FILE_PATH = os.path.join(SAVE_DIR, "wikipedia_restaurant_knowledge.json")

# Load Wikipedia JSON only once at startup
try:
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        wikipedia_data = json.load(file)
    st.sidebar.success(f"✅ Loaded {len(wikipedia_data)} Wikipedia articles.")
except Exception as e:
    st.sidebar.error(f"❌ Error loading Wikipedia data: {e}")

# ------------------------- Chunk Wikipedia Articles -------------------------
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_wikipedia_articles(data, chunk_size=500, chunk_overlap=50):
    """Chunk Wikipedia articles into smaller LangChain Document objects with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    documents = []
    
    for article in data:
        title = article["title"]
        url = article["url"]
        text = article["summary"]
        
        if text:
            split_texts = text_splitter.split_text(text)
            for chunk in split_texts:
                # Convert each chunk into a LangChain Document with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={"title": title, "url": url}  # Metadata for filtering
                )
                documents.append(doc)
    
    return documents

# Convert Wikipedia data to LangChain Documents
chunked_documents = chunk_wikipedia_articles(wikipedia_data)
st.sidebar.success(f"✅ Total chunks created: {len(chunked_documents)}")

# ------------------------- Load Embeddings & FAISS -------------------------

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load Hugging Face Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store Wikipedia text in FAISS (with metadata!)
vector_store = FAISS.from_documents(chunked_documents, embedding_model)

# Save the FAISS index for future use
vector_store.save_local("./wikipedia_faiss_index")

st.sidebar.success(f"✅ Stored {len(chunked_documents)} Wikipedia chunks in FAISS.")



# ------------------------- Define Retrieval Function -------------------------

def retrieve_relevant_chunks(query: str, top_k=5):
    """Retrieve relevant Wikipedia chunks based on user query using LangChain FAISS."""
    retrieved_docs = vector_store.similarity_search(query, k=top_k)

    return retrieved_docs  # Returns a list of LangChain Document objects


# ------------------------- Load Mistral-7B-Instruct Model (GGUF) -------------------------

# Download the GGUF model (only needed once)
model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Load the model once
llm = Llama(model_path=model_path, n_ctx=4096)  # Adjust context length if needed
st.sidebar.success("✅ Mistral-7B-Instruct model loaded successfully.")

# ------------------------- Define Response Generation Function -------------------------

def generate_response(query: str, retrieved_chunks: list) -> str:
    """Generate response using Mistral-7B-Instruct with Wikipedia context."""
    if not retrieved_chunks:
        return "Sorry, I could not find relevant information for your query."

    # Extract text from retrieved LangChain Documents
    context_text = "\n\n".join([f"({i+1}) {doc.page_content}" for i, doc in enumerate(retrieved_chunks)])

    prompt = f"""
    You are an AI assistant answering based on Wikipedia knowledge.

    User Query: {query}

    Relevant Context:
    {context_text}

    Answer the question concisely based on the given context.
    """

    response = llm(prompt, max_tokens=300, temperature=0.7)
    
    return response["choices"][0]["text"]


# ------------------------- Streamlit UI -------------------------

st.title("Wikipedia RAG Chatbot")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        try:
            retrieved_chunks = retrieve_relevant_chunks(query)  # Uses updated retrieval
            answer = generate_response(query, retrieved_chunks)  # Uses updated response function
            st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")

