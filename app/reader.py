"""
This module contains the text extraction and directory reading logic.

Functions:
    extract_text: Compiles the ingested PDF's raw text
    read_directory: Returns the raw text for each PDF in the directory

References:
    - Datastax (Astra) Documentation: https://docs.datastax.com/en/astra-db-serverless/index.html
    - Krish Naik Youtube Channel: https://www.youtube.com/watch?v=zxo3T4aQj6Q
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
"""

from PyPDF2 import PdfReader

def extract_text(pdf):
    """
    Return raw text content from pdf
    """
    reader = PdfReader(pdf)
    raw_text = ''
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def read_directory(directory):
    """
    Returns read text from a directory
    """
    accumulated_text = ""
    for pdf in directory.glob('*.pdf'):
        accumulated_text += extract_text(pdf)
    return accumulated_text
