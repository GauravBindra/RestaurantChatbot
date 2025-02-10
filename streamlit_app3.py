import os
import json
import pandas as pd
import faiss
import numpy as np
import streamlit as st
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ------------------------- Load CSV Data -------------------------
def custom_csv_loader(file_path):
    """
    Converts CSV data into structured text documents with metadata for RAG.
    """
    df = pd.read_csv(file_path)
    documents = []
    
    for _, row in df.iterrows():
        # Construct the textual representation for embedding
        text_representation = f"""
        Restaurant: {row['restaurant_name']}
        Menu Item: {row['menu_item']}
        Category: {row['menu_category']}
        Description: {row['menu_description']}
        Ingredients: {row['ingredients']}
        Price: {row['price_description']}
        Review Summary: {row['review_count_description']}
        Rating Summary: {row['rating_description']}
        Category Description: {row['category_description']}
        """

        # Metadata for filtering
        metadata = {
            "source": "csv",
            "restaurant_name": row["restaurant_name"],
            "location": row["location"],
            "rating": row["rating"],
            "categories": row["category_list"]
        }

        # Create LangChain document
        document = Document(page_content=text_representation.strip(), metadata=metadata)
        documents.append(document)

    return documents

# Load CSV as LangChain Documents
csv_file_path = "./Data/restaurant_data_2.csv"
csv_documents = custom_csv_loader(csv_file_path)
st.sidebar.success(f"✅ Loaded {len(csv_documents)} CSV documents.")

category_list = [
    "Acai Bowls", "American", "Asian Fusion", "Bakeries", "Barbeque",
    "Bars", "Beer Bar", "Bowling", "Brazilian", "Breakfast & Brunch", 
    "Bubble Tea", "Burgers", "Cafes", "Cajun/Creole", "Cantonese", 
    "Caterers", "Cheesesteaks", "Chicken Wings", "Chinese", "Cocktail Bars",
    "Coffee & Tea", "Comfort Food", "Desserts", "Donuts", "Fast Food",
    "Filipino", "Food Delivery Services", "Food Trucks", "French",
    "Gastropubs", "German", "Gluten-Free", "Greek", "Guamanian",
    "Halal", "Hawaiian", "Himalayan/Nepalese", "Hot Dogs", "Indian",
    "Indonesian", "Italian", "Izakaya", "Japanese", "Japanese Curry",
    "Juice Bars & Smoothies", "Kebab", "Kombucha", "Korean",
    "Latin American", "Meat Shops", "Mediterranean", "Mexican", "Music Venues",
    "New American", "Noodles", "Patisserie/Cake Shop", "Persian/Iranian", "Pizza",
    "Poke", "Ramen", "Salad", "Sandwiches", "Seafood", "Shanghainese",
    "Soul Food", "Soup", "Southern", "Spanish", "Specialty Food",
    "Sports Bars", "Sushi Bars", "Tacos", "Tapas Bars", "Tapas/Small Plates",
    "Thai", "Vegan", "Vegetarian", "Venues & Event Spaces", "Vietnamese", "Wine Bars"
]

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
wikipedia_documents = chunk_wikipedia_articles(wikipedia_data)
st.sidebar.success(f"✅ Loaded {len(wikipedia_documents)} Wikipedia documents.")

# ------------------------- Combine Documents -------------------------
all_documents = csv_documents + wikipedia_documents
st.sidebar.success(f"✅ Total documents loaded: {len(all_documents)} (CSV + Wikipedia)")



# ------------------------- Load Embeddings & FAISS -------------------------

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load Hugging Face Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store Wikipedia text in FAISS (with metadata!)
vector_store = FAISS.from_documents(all_documents, embedding_model)

# Save the FAISS index for future use
vector_store.save_local("./wikipedia_faiss_index")

st.sidebar.success(f"✅ Stored {len(all_documents)} Wikipedia chunks in FAISS.")


##
def extract_filters(user_query, file_path):
    """
    Extracts metadata filters from the user query based on dynamically loaded categories, locations, and ratings.
    Reads the dataset from file_path to extract unique category and rating values.
    Works for both structured (CSV) and unstructured (Wikipedia) data.
    """
    df = pd.read_csv(file_path)

    # Extract unique categories and ratings from the dataset
    unique_categories = set()
    df["category_list"].dropna().apply(lambda x: unique_categories.update(eval(x) if isinstance(x, str) else x))
    category_list = sorted(unique_categories)

    unique_ratings = sorted(df["rating"].dropna().unique())

    filters = {}

    # Example predefined locations
    locations = ["Los Angeles", "San Francisco", "New York"]

    # Extract multiple locations
    matching_locations = [loc for loc in locations if loc.lower() in user_query.lower()]
    if matching_locations:
        filters["location"] = matching_locations  # Store as a list

    # Extract multiple categories
    matching_categories = [cat for cat in category_list if cat.lower() in user_query.lower()]
    if matching_categories:
        filters["categories"] = matching_categories  # Store as a list

    # Extract multiple ratings
    matching_ratings = [rating for rating in unique_ratings if str(rating) in user_query]
    if matching_ratings:
        filters["rating"] = matching_ratings  # Store as a list

    # Wikipedia-Specific Filtering (Extracting Topics)
    wikipedia_keywords = ["history", "origin", "traditional", "popular", "famous"]
    matching_wiki_terms = [word for word in wikipedia_keywords if word.lower() in user_query.lower()]
    if matching_wiki_terms:
        filters["source"] = "wikipedia"  # Prioritize Wikipedia content

    return filters

# Example usage:
# filters_applied = extract_filters(query, csv_file_path)
# print(filters_applied)

# ------------------------- Dynamic k Selection -------------------------
def get_dynamic_k(query):
    """
    Determines the value of k dynamically based on query type.
    """
    if "compare" in query or "trend" in query:
        return 20  # Higher k for broad trend-based queries
    elif "find" in query or "list" in query:
        return 10  # Standard k for search queries
    else:
        return 5  # Default for direct lookups


# ------------------------- Boost FAISS Results with Metadata Filtering -------------------------
def boost_faiss_results(results, filters_applied):
    """
    Applies metadata filtering manually for FAISS since it does not support native metadata filtering.
    Instead of removing non-matching results, this boosts the score of matching results.
    """
    boosted_results = []

    for res in results:
        boost_score = 0  # Start with no boost

        for key, value in filters_applied.items():
            # Check if the key exists in metadata
            if key in res.metadata:
                metadata_value = res.metadata[key]

                # Partial match for categories (if stored as a list)
                if key == "categories":
                    if isinstance(metadata_value, list):  # List-based filtering
                        if any(str(val).lower() in str(item).lower() for item in value for val in value):
                            boost_score += 1
                    else:  # Text-based filtering (substring search)
                        if str(value).lower() in str(metadata_value).lower():
                            boost_score += 1

                # Boost rating matches (±0.5 tolerance)
                elif key == "rating":
                    if any(abs(float(metadata_value) - float(val)) <= 0.5 for val in value):
                        boost_score += 1

                # Boost location matches (substring match)
                elif key == "location":
                    if any(str(val).lower() in str(metadata_value).lower() for val in value):
                        boost_score += 1

                # Prioritize Wikipedia when relevant
                elif key == "source" and value == "wikipedia":
                    if metadata_value == "wikipedia":
                        boost_score += 2  # Wikipedia results get a boost

                # Default partial match for other fields
                else:
                    if str(value).lower() in str(metadata_value).lower():
                        boost_score += 1

        # Store the result with its boost score
        boosted_results.append((boost_score, res))

    # Sort results based on the boost score (higher is better)
    boosted_results.sort(reverse=True, key=lambda x: x[0])

    # Extract the sorted documents
    sorted_results = [res for _, res in boosted_results]

    return sorted_results



# ------------------------- Define Retrieval Function -------------------------

# ------------------------- Modify Retrieval Function -------------------------
# def retrieve_relevant_chunks(query: str, top_k=5):
#     """
#     Retrieves relevant chunks from the FAISS index and applies metadata-based filtering.
#     """
#     filters_applied = extract_filters(query, csv_file_path)  # Extract filters
#     k = get_dynamic_k(query)  # Adjust retrieval count based on query type

#     results = vector_store.similarity_search(query, k=k)  # Retrieve initial results
#     boosted_results = boost_faiss_results(results, filters_applied)  # Apply metadata boosting

#     return boosted_results

def retrieve_relevant_chunks(query: str, top_k=5):
    """
    Retrieves relevant chunks from FAISS and prioritizes structured data for restaurant-related queries.
    """
    filters_applied = extract_filters(query, csv_file_path)
    k = get_dynamic_k(query)

    # Retrieve raw FAISS results
    results = vector_store.similarity_search(query, k=k)
    
    # Apply strict filtering for restaurant-related queries
    if any(keyword in query.lower() for keyword in ["restaurant", "menu", "dish", "food", "serve"]):
        # Only keep structured (CSV) results and boost them
        results = [doc for doc in results if doc.metadata.get("source") == "csv"]

    boosted_results = boost_faiss_results(results, filters_applied)
    return boosted_results


# ------------------------- Load Mistral-7B-Instruct Model (GGUF) -------------------------

# Download the GGUF model (only needed once)
model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Load the model once
llm = Llama(model_path=model_path, n_ctx=4096)  # Adjust context length if needed
st.sidebar.success("✅ Mistral-7B-Instruct model loaded successfully.")

# ------------------------- Modify Response Generation Function -------------------------
def generate_response(query: str, retrieved_chunks: list) -> str:
    """
    Generates a response using Mistral-7B-Instruct with structured and unstructured context.
    """
    if not retrieved_chunks:
        return "Sorry, I could not find relevant information for your query."

    context_text = "\n\n".join([
        f"({i+1}) [{doc.metadata.get('source', 'unknown')}] {doc.page_content}" 
        for i, doc in enumerate(retrieved_chunks)
    ])

    # prompt = f"""
    # You are an AI assistant answering based on restaurant knowledge.

    # User Query: {query}

    # Relevant Context:
    # {context_text}

    # Answer the question concisely based on the given context.
    # """

#     prompt = f"""
# You are an AI assistant specializing in restaurant-related queries. Your responses must be strictly based on the provided information.

# ### **Context Details**
# The retrieved documents come from two knowledge sources:
# 1. **Structured Restaurant Database (CSV-based)** – Includes restaurant names, menu items, pricing, reviews, and ingredients.
# 2. **Unstructured Wikipedia Knowledge (JSON-based)** – Includes restaurant history, food trends, Michelin Guide information, and general culinary knowledge.

# ### **Guidelines for Answering:**
# - **Use only the retrieved context**. Do not make up information.
# - **Prioritize structured restaurant data** for factual queries (e.g., menu items, pricing, reviews).
# - Use **Wikipedia knowledge** when answering general questions (e.g., history, trends, Michelin Guide).
# - **Consider metadata** (location, rating, categories) in responses to provide precise answers.
# - If **insufficient data exists**, say: "I don't have enough information to answer this."
# - **Be concise** and summarize where needed.

# ### **Retrieved Context:**  
# {context_text}

# ### **User Query:**  
# {query}

# ### **Answer:**
# """
    prompt = f"""
### 🍽️ AI Assistant for Restaurant & Culinary Queries  

You are an expert AI assistant specializing in restaurant-related queries. Your responses **must be strictly based on the retrieved information**.  

---

### **📖 Context Details**
The retrieved documents come from **two structured knowledge sources**:  
1️⃣ **Structured Restaurant Database (CSV-based)** – Includes **restaurant names, menu items, pricing, reviews, and ingredients**.  
2️⃣ **Unstructured Wikipedia Knowledge (JSON-based)** – Includes **restaurant history, food trends, Michelin Guide information, and general culinary knowledge**.  

---

### **📝 Guidelines for Answering:**
✅ **Strictly use the retrieved context**. If the requested information **is not found, say:**  
   ❌ *"I don't have enough information to answer this."*  

✅ **For factual restaurant-related questions** (menu items, pricing, reviews, locations):  
   - Use **only the structured restaurant dataset**.
   - **Do not make up restaurant names, dishes, or locations**.

✅ **For general knowledge questions** (e.g., food trends, Michelin Guide, history):  
   - Use **Wikipedia knowledge** when relevant.  
   - Clearly indicate if an answer comes from Wikipedia.  

✅ **Use metadata (location, rating, categories)** to provide precise and contextualized responses.  

✅ **Summarize long responses** and avoid unnecessary details.  

---

### **🔎 Retrieved Context:**  
{context_text}

### **❓ User Query:**  
{query}

### **📝 Answer:**  
"""

    response = llm(prompt, max_tokens=300, temperature=0.7)
    
    return response["choices"][0]["text"]


# ------------------------- Modify Streamlit UI -------------------------
st.title("Unified RAG Chatbot (CSV + Wikipedia)")

# query = st.text_input("Ask a question:")

# if st.button("Submit"):
#     if query:
#         try:
#             retrieved_chunks = retrieve_relevant_chunks(query)  # Uses updated retrieval
#             answer = generate_response(query, retrieved_chunks)  # Uses updated response function
#             st.write(f"**Answer:** {answer}")
#         except Exception as e:
#             st.error(f"Error: {e}")
#     else:
#         st.warning("Please enter a question.")

# st.title("Restaurant & Food Chatbot")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        try:
            retrieved_chunks = retrieve_relevant_chunks(query)
            
            if not retrieved_chunks:
                st.write("⚠️ I could not find relevant information in my database.")

            else:
                answer = generate_response(query, retrieved_chunks)
                st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
