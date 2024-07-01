"""
This module contains the main execution logic used by the LLM 
application on Streamlit.

Functions:
    setup_environment: Initializes the environment variables and LLM components
    process_documents: Reads and processes the documents to create the vector index
    process_question: Processes the user queries to return the appropriate response

References:
    - Datastax (Astra) Documentation: https://docs.datastax.com/en/astra-db-serverless/index.html
    - Krish Naik Youtube Channel: https://www.youtube.com/watch?v=zxo3T4aQj6Q
    - OpenAI API Documentation: https://platform.openai.com/docs/introduction
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - Langchain Documentation: https://python.langchain.com/v0.2/docs/integrations/llms/openai/
"""

from .utils import get_env_var
from .components import initialize_cassio, llm_embedding
from .reader import read_directory
from .splitter import intelligent_chunk
from .vector import create_vector_store, load_text, index_text
from .lengths import analyze_length

def setup_environment():
    """
    Set up the environment and initialize components.
    """
    initialize_cassio(get_env_var('ASTRA_TOKEN'), get_env_var('ASTRA_DB_ID'))
    llm, embedding = llm_embedding(get_env_var('OPENAI_API_KEY'))
    return llm, embedding

def process_documents(directory, embedding):
    """
    Read, chunk, and process documents.
    """
    # Read documents
    text = read_directory(directory)

    # Chunk documents
    mean_length, std_length, max_length = analyze_length(directory)
    possible_chunk = int(mean_length + std_length)
    model_max = 4097
    buffer = 100
    max_chunk_size = min(possible_chunk, max_length / 2, model_max - buffer)
    splitted_text = intelligent_chunk(text, max_chunk_size=max_chunk_size)

    # Process documents
    vector_store = create_vector_store(embedding, table_name="sumhack")
    vector_store = load_text(vector_store, splitted_text)
    vector_index = index_text(vector_store)
    
    return vector_index

def process_question(query, vector_index, llm):
    """
    Process a given query and return the response.
    """
    response = vector_index.query(query, llm).strip()
    return response
