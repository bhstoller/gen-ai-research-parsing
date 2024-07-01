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

import streamlit as st
from pathlib import Path
from app.main import setup_environment, process_documents, process_question

# Cache the environment setup
@st.cache(allow_output_mutation=True)
def cached_setup_environment():
    return setup_environment()

# Initialize components and environment
llm, embedding = cached_setup_environment()

# Cache the document processing
@st.cache(allow_output_mutation=True)
def cached_process_documents(directory):
    return process_documents(directory, embedding)

# Process research documents
directory = Path("documents")
document_files = [f for f in directory.glob('*.pdf')]
document_names = ["Select a Document"] + [doc.name for doc in document_files]

# Create the vector index
vector_index = cached_process_documents(directory)

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
        try:
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
        except Exception as e:
            st.error("An error occurred while processing your request. Please try again.")
    else:
        # Handle incorrect submission
        st.write("Please enter a valid query.")

# Prepare summary Streamlit User Interface (UI)
st.write("---")
st.write("Request summaries of the research documents")

# Show dropdown menu for document selection
selected_document = st.selectbox("Select a document to summarize:", document_names, index=0)

# Initialize empty placeholder for the progress bar and summary response
summary_progress_bar_placeholder = st.empty()
summary_response_placeholder = st.empty()

# Summary Submission Logic
if st.button("Summarize Document"):
    if selected_document != "Select a Document":
        try:
            # Clear previous summary
            summary_response_placeholder.empty()
            
            # Query LLM for the summary
            summary_query = (f"Summarize {selected_document}. Include the methodology, techniques, and conclusion")
            
            # Initialize the progress bar
            summary_progress_bar = summary_progress_bar_placeholder.progress(0)
            
            # Show progress
            for i in range(100):
                import time
                time.sleep(0.05)
                summary_progress_bar.progress(i + 1)
            
            # Load summary response
            summary_response = process_question(summary_query, vector_index, llm)
            summary_response_placeholder.write("Summary:")
            summary_response_placeholder.markdown(summary_response)
            
            # Complete and remove progress bar
            summary_progress_bar.progress(100)
            summary_progress_bar_placeholder.empty()
        except Exception as e:
            st.error("An error occurred while summarizing the document. Please try again.")
    else:
        # Handle incorrect submission
        st.write("Please select a document to summarize.")
