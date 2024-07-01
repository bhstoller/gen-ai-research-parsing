"""
This module contains the vector database operations logic, which is
hosted by Astra/Cassandra on Datastax.

Functions:
    create_vector_store: Creates the Astra vector store
    load_text: Loads the text into vectors inside the vector store
    index_text: Indexes the vectors inside the vector store

References:
    - Datastax (Astra) Documentation: https://docs.datastax.com/en/astra-db-serverless/index.html
    - Krish Naik Youtube Channel: https://www.youtube.com/watch?v=zxo3T4aQj6Q
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - Langchain Documentation: https://python.langchain.com/v0.2/docs/integrations/llms/openai/
"""

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Cassandra

def create_vector_store(embedding, table_name, session=None, keyspace=None):
    """
    Return the created Astra vector store
    """
    astra_vector_store = Cassandra(
        embedding= embedding,
        table_name= table_name,
        session= session,
        keyspace= keyspace
    )
    return astra_vector_store

def load_text(vector_store, texts):
    """
    Return the updated Astra vector store after laoding
    the text into the vectors
    """
    vector_store.add_texts(texts)
    return vector_store

def index_text(vector_store):
    """
    Return the vector index of the Astra vector store
    """
    vector_index = VectorStoreIndexWrapper(vectorstore= vector_store)
    return vector_index
