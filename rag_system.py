import pandas as pd
from langchain.schema import Document

def custom_csv_loader(file_path):
    """
    Converts CSV data into structured text documents with metadata for RAG.
    Processes grouped menu items to ensure each menu item is a single document.
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

        # Metadata for filtering (restaurant_name removed)
        metadata = {
            "location": row["location"],
            "rating": row["rating"],
            "categories": row["category_list"]#,  # Stored as a list for filtering
            # "ingredients": row["ingredients"].split(", ")  # Store as a list for ingredient filtering
        }

        # Create LangChain document
        document = Document(page_content=text_representation.strip(), metadata=metadata)
        documents.append(document)

    return documents



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

file_path = "./Data/restaurant_data_2.csv"
documents = custom_csv_loader(file_path)

# Display first document for verification
print(documents[0])

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS for efficient retrieval
vector_store = FAISS.from_documents(documents, embedding_model)

# Save FAISS index for later use
vector_store.save_local("./faiss_index")

# Load FAISS index
vector_store = FAISS.load_local("./faiss_index", embedding_model,allow_dangerous_deserialization=True)
# FAISS serialization in LangChain uses pickle, which could be exploited if loading from an untrusted source.

# Query example
query = "What are the top-5 trending ingredients in mexican restaurants"

import pandas as pd


def extract_filters(user_query, file_path):
    """
    Extracts metadata filters from the user query based on dynamically loaded categories, locations, and ratings.
    Reads the dataset from file_path to extract unique category and rating values.
    Ensures multiple matches are extracted for better filtering.
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

    return filters

filters_applied = extract_filters(query, file_path)
print(filters_applied)

def get_dynamic_k(query):
    if "compare" in query or "trend" in query:
        return 20  # Higher k for broad trend-based queries
    elif "find" in query or "list" in query:
        return 10  # Standard k for search queries
    else:
        return 5  # Default for direct lookups
k = get_dynamic_k(query)
print(k)

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
                        if any(str(value).lower() in str(item).lower() for item in metadata_value):
                            boost_score += 1
                    else:  # Text-based filtering (substring search)
                        if str(value).lower() in str(metadata_value).lower():
                            boost_score += 1

                # Boost rating matches (±0.5 tolerance)
                elif key == "rating":
                    if value - 0.5 <= metadata_value <= value + 0.5:
                        boost_score += 1

                # Boost location matches (substring match)
                elif key == "location":
                    if str(value).lower() in str(metadata_value).lower():
                        boost_score += 1

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

    # Display the results along with metadata and boost score
    for boost, res in boosted_results:
        print(f"Boost Score: {boost}")
        print(f"Restaurant: {res.metadata.get('restaurant_name', 'Unknown')}")
        print(f"Menu Item: {res.page_content}")
        print(f"Metadata: {res.metadata}")
        print("-" * 50)

    return sorted_results

# Example Query Execution (Boosting Instead of Removing)
results = vector_store.similarity_search(query, k=k)  # Retrieve first
boosted_results = boost_faiss_results(results, filters_applied)


print(len(boosted_results))

for res in results:
        print(f"Restaurant: {res.metadata.get('restaurant_name', 'Unknown')}")
        print(f"Menu Item: {res.page_content}")
        print(f"Metadata: {res.metadata}")
        print("-" * 50)

context = "\n\n".join([doc.page_content for doc in filtered_results])
prompt = f"""
You are a helpful assistant. Answer the question using the provided information.

Context:
{context}

Question: {query}
Answer:
"""

from huggingface_hub import hf_hub_download

# Replace with the exact filename from the GGUF model page
model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

print("Model path:", model_path)

from llama_cpp import Llama

# ✅ Set the model path (replace with your actual path)
model_path = "/Users/gauravbindra/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# ✅ Load model with optimized CPU settings
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=6)  # Use 6 threads for your 6-core CPU

# ✅ Test inference
# query = "What is the capital of France?"
# response = llm(f"Answer the following question:\n{query}")

# # ✅ Print the response
# print(response["choices"][0]["text"])

response = llm(prompt, max_tokens=256) 
print("AI Response:", response["choices"][0]["text"])

response = llm(prompt, max_tokens=512) 
print("AI Response:", response["choices"][0]["text"])



