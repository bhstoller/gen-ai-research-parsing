"""
This module contains the main execution logic for the LLM project.

It includes functions to read documents, chunk text, create Pinecone index,
embed and store documents in the Pinecone index, and query the index.
"""

from pathlib import Path
from app.utils import get_env_var
from app.reader import read_directory
from app.splitter import text_splitter, split_text
from app.components import initialize_cassio, create_embedding, create_vector_store, load_text, load_index

astra_id = get_env_var('ASTRA_DB_ID')
astra_token = get_env_var('ASTRA_TOKEN')
openai_api_key = get_env_var('OPENAI_API_KEY')

initialize_cassio(astra_token, astra_id)
llm, embedding = create_embedding(openai_api_key)
vector_store = create_vector_store(embedding, table_name= "sumhack")

directory = Path("documents")
raw_text = read_directory(directory)

splitter = text_splitter()
splitted_text = split_text(raw_text, splitter)

vector_store = load_text(vector_store, splitted_text)
vector_index = load_index(vector_store)

def main():
    """
    Run main
    """
    while True:
        query = input("Question: ")
        if query.lower() == 'quit' or query == "":
            break
        else:
            response = vector_index.query(query, llm).strip()
            print("Answer:", response)

if __name__ == "__main__":
    main()
