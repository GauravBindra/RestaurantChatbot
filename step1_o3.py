import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

###########################
# 1. Data Ingestion
###########################

# Function to ingest proprietary restaurant data from a CSV file.
def load_restaurant_data(csv_path):
    """
    Reads the CSV containing restaurant data and converts each row into a descriptive text passage.
    Expected CSV columns: 'restaurant_name', 'city', 'dishes', 'description'
    """
    df = pd.read_csv(csv_path)
    passages = []
    metadata = []  # To store additional info for later reference.
    for idx, row in df.iterrows():
        text = (
            f"Restaurant: {row['restaurant_name']}. "
            f"City: {row['city']}. "
            f"Dishes: {row['dishes']}. "
            f"Description: {row.get('description', '')}"
        )
        passages.append(text)
        metadata.append({
            'source': 'restaurant_csv',
            'id': idx,
            'restaurant_name': row['restaurant_name']
        })
    return passages, metadata

# Function to ingest unstructured external documents from a folder.
def load_external_docs(folder_path):
    """
    Reads all .txt files from the specified folder and returns their contents as text passages.
    """
    passages = []
    metadata = []
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        passages.append(content)
        metadata.append({
            'source': 'external_doc',
            'file_path': filepath
        })
    return passages, metadata

# Ingest data from both sources.
csv_passages, csv_metadata = load_restaurant_data("restaurant_data.csv")
external_passages, external_metadata = load_external_docs("external_docs")

# Combine all passages and metadata into single lists.
all_passages = csv_passages + external_passages
all_metadata = csv_metadata + external_metadata

###########################
# 2. Vectorization (Embedding)
###########################

# Initialize the embedding model.
# Using 'all-MiniLM-L6-v2' for its balance of speed and performance.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all text passages.
# The 'show_progress_bar=True' flag provides visual feedback during encoding.
embeddings = embedder.encode(all_passages, show_progress_bar=True, convert_to_numpy=True)

###########################
# 3. Indexing with FAISS
###########################

# Determine the dimensionality of the embeddings.
embedding_dim = embeddings.shape[1]

# Create a FAISS index. Here, we use IndexFlatL2 (a brute-force, exact search index).
# It's simple and effective for smaller datasets.
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add the embeddings to the FAISS index.
faiss_index.add(embeddings)

###########################
# 4. Testing the Setup
###########################

# For demonstration, let's perform a test query.
query = "Looking for restaurants serving vegan options in Los Angeles"
query_embedding = embedder.encode([query], convert_to_numpy=True)
distances, indices = faiss_index.search(query_embedding, k=5)

print("Top 5 similar passages for the query:")
for idx in indices[0]:
    print("Passage:")
    print(all_passages[idx])
    print("Metadata:", all_metadata[idx])
    print("------")
