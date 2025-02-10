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

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_wikipedia_articles(data, chunk_size=500, chunk_overlap=50):
    """Chunk Wikipedia articles into smaller segments for efficient retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    
    for article in data:
        title = article["title"]
        url = article["url"]
        text = article["summary"]
        
        if text:
            split_texts = text_splitter.split_text(text)
            for chunk in split_texts:
                chunks.append({"title": title, "url": url, "chunk": chunk})
    
    return chunks

# Chunk the articles (Do this once)
chunked_data = chunk_wikipedia_articles(wikipedia_data)
st.sidebar.success(f"✅ Total chunks created: {len(chunked_data)}")

# ------------------------- Load Embeddings & FAISS -------------------------

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings (Only once)
chunks_texts = [chunk["chunk"] for chunk in chunked_data]
vectors = embedding_model.encode(chunks_texts, convert_to_numpy=True)

# Store in FAISS for fast retrieval
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

st.sidebar.success(f"✅ Stored {len(vectors)} chunks in FAISS.")

# ------------------------- Define Retrieval Function -------------------------

def retrieve_relevant_chunks(query: str, top_k=5):
    """Retrieve relevant Wikipedia chunks based on user query."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        results.append(chunked_data[i])
    
    return results

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

    context_text = "\n\n".join([f"({i+1}) {chunk['chunk']}" for i, chunk in enumerate(retrieved_chunks)])
    
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
            retrieved_chunks = retrieve_relevant_chunks(query)
            answer = generate_response(query, retrieved_chunks)
            st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
