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

import os
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Cassandra
from app.utils import get_env_var, tokenize_sentence
from app.components import initialize_cassio, llm_embedding
from app.reader import read_directory
from app.splitter import intelligent_chunk
from app.vector import create_vector_store, load_text, index_text
from app.lengths import analyze_length

def setup_environment():
    """
    Initializes the environment variables and LLM components.
    """
    astra_id = get_env_var('ASTRA_DB_ID')
    astra_token = get_env_var('ASTRA_TOKEN')
    openai_api_key = get_env_var('OPENAI_API_KEY')

    initialize_cassio(astra_token, astra_id)
    llm, embedding = llm_embedding(openai_api_key)
    return llm, embedding

def process_documents(directory, embedding):
    """
    Reads and processes documents to create the vector index.
    """
    mean_length, median_length, max_length = analyze_length(directory)
    max_chunk_size = int(mean_length / 2)
    model_max_content_length = 4097
    buffer_length = 100
    max_chunk_size = min(max_chunk_size, model_max_content_length - buffer_length)

    text = read_directory(directory)
    splitted_text = intelligent_chunk(text, max_chunk_size=max_chunk_size)

    vector_store = create_vector_store(embedding, table_name="sumhack")
    vector_store = load_text(vector_store, splitted_text)
    vector_index = index_text(vector_store)
    return vector_index

def process_question(query, vector_index, llm):
    """
    Processes the user queries to return the appropriate response.
    """
    response = vector_index.query(query, llm).strip()
    return response
