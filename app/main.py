"""
This module contains the main execution logic for the LLM project.

It includes functions to read documents, chunk text, create Pinecone index,
embed and store documents in the Pinecone index, and query the index.
"""

from pathlib import Path
from .utils import get_env_var
from .components import initialize_cassio, llm_embedding
from .reader import read_directory
from .splitter import split_text
from .vector import create_vector_store, load_text, index_text

astra_id = get_env_var('ASTRA_DB_ID')
astra_token = get_env_var('ASTRA_TOKEN')
openai_api_key = get_env_var('OPENAI_API_KEY')

initialize_cassio(astra_token, astra_id)
llm, embedding = llm_embedding(openai_api_key)

directory = Path("documents")
text = read_directory(directory)
splitted_text = split_text(text, chunk_size=3200)

vector_store = create_vector_store(embedding, table_name="sumhack")
vector_store = load_text(vector_store, splitted_text)
vector_index = index_text(vector_store)

def process_question(query):
    """
    Process a given query and return the response.
    """
    response = vector_index.query(query, llm).strip()
    return response

def main():
    """
    Run main
    """
    while True:
        query = input("Question: ")
        if query.lower() == 'quit' or query == "":
            break
        response = process_question(query)
        print("Answer:", response)

if __name__ == "__main__":
    main()
