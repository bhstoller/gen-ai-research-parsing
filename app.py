import streamlit as st
from pathlib import Path
from app.utils import get_env_var
from app.components import initialize_cassio, llm_embedding
from app.reader import read_directory
from app.splitter import split_text
from app.vector import create_vector_store, load_text, index_text, clear_vector_store
from langchain_openai import OpenAI, OpenAIEmbeddings

# Initialize environment variables
astra_id = get_env_var('ASTRA_DB_ID')
astra_token = get_env_var('ASTRA_TOKEN')
openai_api_key = get_env_var('OPENAI_API_KEY')

# Initialize components
initialize_cassio(astra_token, astra_id)
llm, embedding = llm_embedding(openai_api_key)

# Read and process documents
directory = Path("documents")
text = read_directory(directory)
splitted_text = split_text(text, chunk_size=3200)

# Create and load vector store
vector_store = create_vector_store(embedding, table_name="sumhack")

# Clear existing data
vector_store = clear_vector_store(vector_store)

try:
    vector_store = load_text(vector_store, splitted_text)
    vector_index = index_text(vector_store)
except Exception as e:
    st.error(f"Error loading text into vector store: {e}")
    raise e

def process_question(query):
    """
    Process a given query and return the response.
    """
    try:
        response = vector_index.query(query, llm).strip()
        return response
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return "An error occurred while processing your question."

# Streamlit app layout
st.title("Research Document Query Application")
st.write("Ask questions about the research documents:")

# User input
user_input = st.text_input("Enter your question here:")

# Submit button
if st.button("Submit"):
    if user_input:
        response = process_question(user_input)
        st.write(response)
    else:
        st.write("Please enter a valid query.")
