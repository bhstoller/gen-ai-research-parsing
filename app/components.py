"""
This module contains the vector database and LLM intiailization logic. The vector
database is hosted by Astra/Cassandra, while the LLM and embedding
uses OpenAI's GPT-3 model.

Functions:
    initialize_cassio: Initializes the vector database 
    llm_embedding: Generates the GPT-3 LLM and embedding

References:
    - Datastax (Astra) Documentation: https://docs.datastax.com/en/astra-db-serverless/index.html
    - OpenAI API Documentation: https://platform.openai.com/docs/introduction
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - Krish Naik Youtube Channel: https://www.youtube.com/watch?v=zxo3T4aQj6Q
    - Langchain Documentation: https://python.langchain.com/v0.2/docs/integrations/llms/openai/
"""


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import cassio

def initialize_cassio(db_token, db_id):
    """
    Initialize the Astra/Cassandra vector database
    """
    cassio.init(token=db_token, database_id=db_id)

def llm_embedding(api_key):
    """
    Return the instantiated GPT-4o LLM and embedding
    """
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o",
        temperature=0
    )
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    return llm, embedding
