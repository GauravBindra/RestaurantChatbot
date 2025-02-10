import streamlit as st
import requests

# Set your Cloudflare Tunnel API URL here
API_URL = "https://oxford-periodic-mistakes-displays.trycloudflare.com"

st.title("Wikipedia RAG Chatbot")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        try:
            response = requests.get(API_URL, params={"query": query}, timeout=30)
            response.raise_for_status()  # Raise an error for HTTP issues (4xx, 5xx)
            answer = response.json().get("answer", "No response")
            st.write(f"**Answer:** {answer}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
