"""
This module contains the PDF reading logic.

It includes functions read and parse the text in the PDFs.
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
