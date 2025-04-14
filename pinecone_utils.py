import os
import uuid
import streamlit as st
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Function to get API keys from environment or Streamlit secrets
def get_api_key(key_name):
    # First try environment variables
    value = os.getenv(key_name)
    
    # If not found, try Streamlit secrets
    if not value and 'secrets' in dir(st):
        try:
            value = st.secrets[key_name]
        except:
            pass
            
    return value

# Get necessary API keys
PINECONE_API_KEY = get_api_key("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_api_key("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)

# Store chat into Pinecone
def store_chat(question, response):
    """Store chat question and response in Pinecone."""
    try:
        combined_text = f"Question: {question}\nAnswer: {response}"
        vector = embedding_model.embed_query(combined_text)
        
        metadata = {"question": question, "response": response}
        vector_id = str(uuid.uuid4())

        index.upsert(vectors=[{
            "id": vector_id,
            "values": vector,
            "metadata": metadata
        }])
        return True
    except Exception as e:
        print(f"Error storing chat: {str(e)}")
        return False

# Retrieve similar chats from Pinecone
def search_similar(query, top_k=3):
    """Search for similar chats in Pinecone."""
    try:
        query_vector = embedding_model.embed_query(query)
        search_result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        results = []
        for match in search_result['matches']:
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            results.append({
                "question": metadata.get("question", ""),
                "response": metadata.get("response", ""),
                "score": score
            })

        return results
    except Exception as e:
        print(f"Error searching similar chats: {str(e)}")
        return []
