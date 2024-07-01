"""
This module contains the Streamlit application logic for querying and 
summarizing the directory of research documents using the LLM application.

The application initializes the environment, processes documents, and allows users to 
submit queries or request summaries of the documents.

Usage:
    - The user can enter a question in the text input to query the processed documents.
    - The user can select a document from the dropdown menu and click the "Summarize Document" 
      button to get a summary of the selected document.

References:
    - Streamlit Documentation: https://docs.streamlit.io
    - OpenAI API Documentation: https://platform.openai.com/docs/introduction
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - Langchain Documentation: https://python.langchain.com/v0.2/docs/integrations/llms/openai/
    - Real Python Article: https://realpython.com/build-llm-rag-chatbot-with-langchain/
"""

from pathlib import Path
import streamlit as st
from app.main import setup_environment, process_documents, process_question

# Initialize components and environment
llm, embedding = setup_environment()

# Process research documents
directory = Path("documents")
document_files = [f for f in directory.glob('*.pdf')]
document_names = ["Select a Document"] + [doc.name for doc in document_files]

# Create the vector index
vector_index = process_documents(directory, embedding)

# Prepare the query/answer Streamlit User Interface (UI)
st.title("Research Document Query Application")
st.write("Ask questions about the research documents")
user_input = st.text_input("Enter your question here:")

# Initialize empty placeholder for the progress bar and query response
progress_bar_placeholder = st.empty()
response_placeholder = st.empty()

# Query Submission Logic
if st.button("Submit"):
    if user_input:
        # Clear previous response
        response_placeholder.empty()
        
        # Initialize the progress bar
        progress_bar = progress_bar_placeholder.progress(0)
        
        # Show progress
        for i in range(100):
            import time
            time.sleep(0.05)
            progress_bar.progress(i + 1)
        
        # Load query response
        response = process_question(user_input, vector_index, llm)
        response_placeholder.markdown(response)
        
        # Complete and remove progress bar
        progress_bar.progress(100)
        progress_bar_placeholder.empty()
    else:
        # Handle incorrect submission
        st.write("Please enter a valid query.")

# Prepare summary Streamlit User Interface (UI)
st.write("---")
st.write("Request summaries of the research documents")

# Show dropdown menu for document selection
select
