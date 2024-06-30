"""
This module contains the vector operations logic.

It inludes functions to create the vector storage, load the text, 
index the text, and reset the index.
"""
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Cassandra

def create_vector_store(embedding, table_name, session=None, keyspace=None):
    """
    Create vector store
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
    Load the text into astra vectors
    """
    vector_store.add_texts(texts)
    return vector_store

def load_images(vector_store, image_features):
    """
    Load the images into astra vectors
    """
    vector_store.add_texts(image_features)
    return vector_store

def index_text(vector_store):
    """
    Load the vector index
    """
    vector_index = VectorStoreIndexWrapper(vectorstore= vector_store)
    return vector_index

def clear_vector_store(vector_store):
    """
    Return the vector store after clearing its contents.
    """
    vector_store = vector_store.clear()
    return vector_store
