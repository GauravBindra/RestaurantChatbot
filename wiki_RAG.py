import json
import os

# Load Wikipedia knowledge base
SAVE_DIR = os.path.join("Data", "Wikipedia_data2")
FILE_PATH = os.path.join(SAVE_DIR, "wikipedia_restaurant_knowledge.json")

with open(FILE_PATH, "r", encoding="utf-8") as file:
    wikipedia_data = json.load(file)

print(f"Loaded {len(wikipedia_data)} Wikipedia articles.")


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

# Chunk the articles
chunked_data = chunk_wikipedia_articles(wikipedia_data)
print(f"Total chunks created: {len(chunked_data)}")


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
chunks_texts = [chunk["chunk"] for chunk in chunked_data]
vectors = embedding_model.encode(chunks_texts, convert_to_numpy=True)

# Convert to FAISS format
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

print(f"Stored {len(vectors)} chunks in FAISS.")

def retrieve_relevant_chunks(query, index, chunked_data, top_k=3):
    """Retrieve relevant Wikipedia chunks based on user query."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        results.append(chunked_data[i])
    
    return results

# Example Query
query = "What is the history of sushi, and which restaurants in my area are known for it?"
retrieved_chunks = retrieve_relevant_chunks(query, index, chunked_data)
for chunk in retrieved_chunks:
    print(f"Title: {chunk['title']}, URL: {chunk['url']}\nChunk: {chunk['chunk']}\n")


from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Download the GGUF model (only needed once)
model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Load the model
llm = Llama(model_path=model_path, n_ctx=2048)  # Adjust context length if needed

print("Mistral-7B-Instruct model loaded successfully.")


def generate_response(query, retrieved_chunks):
    """Generate response using Mistral-7B-Instruct with Wikipedia context."""
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

# Generate Answer
answer = generate_response(query, retrieved_chunks)
print("AI Response:", answer)

from fastapi import FastAPI
import uvicorn

# Initialize FastAPI app
app = FastAPI()

@app.get("/ask")
def ask_bot(query: str):
    """API endpoint to handle user queries."""
    retrieved_chunks = retrieve_relevant_chunks(query, index, chunked_data)
    answer = generate_response(query, retrieved_chunks)
    return {"query": query, "answer": answer}

if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
