import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load CSV data and preprocess it.
df = pd.read_csv("restaurant_data.csv")  # Update the file path as needed

# Function to convert a CSV row into a descriptive text passage.
def create_text(row):
    # Adjust the column names to match those in your CSV.
    text = (
        f"Restaurant: {row['restaurant_name']}. "
        f"City: {row['city']}. "
        f"Dishes: {row['dishes']}. "
        f"Description: {row.get('description', '')}"
    )
    return text

# Convert each row into a text passage.
passages = df.apply(create_text, axis=1).tolist()

# 2. Create embeddings for the passages using SentenceTransformer.
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(passages, convert_to_numpy=True)

# 3. Build a FAISS index for efficient vector similarity search.
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings)

# Retrieval function: Given a query, returns the top k most relevant passages.
def retrieve(query, k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    # Retrieve the corresponding passages.
    results = [passages[i] for i in indices[0] if i < len(passages)]
    return results

# 4. Initialize DeepSeek R1 via Hugging Face Transformers.
# Note: Ensure that "deepseek/r1" is the correct model identifier.
model_name = "deepseek/r1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generation function: Uses DeepSeek R1 to generate an answer based on the query and retrieved context.
def generate_answer(query, context_passages):
    # Combine retrieved passages into a single context text.
    context_text = "\n".join(context_passages)
    prompt = (
        f"Using the following context information about restaurants:\n{context_text}\n\n"
        f"Answer the following question:\n{query}\n"
    )
    
    # Generate an answer using DeepSeek R1.
    output = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    answer = output[0]['generated_text']
    return answer

# Function that ties together retrieval and generation.
def answer_question(query):
    retrieved_context = retrieve(query, k=5)
    answer = generate_answer(query, retrieved_context)
    return answer

# Example usage:
if __name__ == "__main__":
    query1 = "Which restaurants in Los Angeles offer dishes with Impossible Meat?"
    answer1 = answer_question(query1)
    print("Answer 1:")
    print(answer1)
    print("\n" + "="*50 + "\n")
    
    query2 = "Give me a summary of the latest trends around desserts in San Francisco."
    answer2 = answer_question(query2)
    print("Answer 2:")
    print(answer2)