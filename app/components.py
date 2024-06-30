"""
This module initializes the vector DB and LLM.

It contains functions to initialize cassio and OpenAI
"""


# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAI, OpenAIEmbeddings
import cassio

def initialize_cassio(db_token, db_id):
    """
    Initialize cassio DB
    """
    cassio.init(token= db_token, database_id= db_id)

def llm_embedding(api_key):
    """
    Generate llm and embedding
    """
    llm = OpenAI(api_key= api_key)
    embedding = OpenAIEmbeddings(api_key= api_key)
    return llm, embedding

# def create_vector_store(embedding, table_name, session=None, keyspace=None):
#     """
#     Create vector store
#     """
#     astra_vector_store = Cassandra(
#         embedding= embedding,
#         table_name= table_name,
#         session= session,
#         keyspace= keyspace
#     )
#     return astra_vector_store

# def load_text(vector_store, texts):
#     """
#     Load the specified number of headlines
#     """
#     vector_store = reset_vector_store(vector_store)
#     vector_store.add_texts(texts)
#     return vector_store

# def load_index(vector_store):
#     """
#     Load the vector index
#     """  
#     vector_index = VectorStoreIndexWrapper(vectorstore= vector_store)
#     return vector_index

# def reset_vector_store(vector_store):
#     """
#     Return the vector store after clearing its contents.
#     """
#     # Assuming vector_store has a method to clear all vectors
#     vector_store.clear()
#     return vector_store