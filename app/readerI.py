"""
This module contains the PDF reading logic.

It includes functions read and parse the text in the PDFs.
"""

import io
import fitz
from PyPDF2 import PdfReader
from PIL import Image
from pathlib import Path

def extract_text(pdf):
    """
    Return raw text content from pdf
    """
    text_reader = PdfReader(pdf)
    extracted_text = ''
    for page in text_reader.pages:
        content = page.extract_text()
        if content:
            extracted_text += content
    return extracted_text

def extract_images(pdf):
    """
    Returns extracted images
    """
    image_file = fitz.open(pdf)
    extracted_images = []
    for page in image_file:
        for image in page.get_images(full= True):
            xref = image[0]
            base_image = image_file.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            extracted_images.append(image)
    image_file.close()
    return extracted_images

def read_directory(directory):
    """
    Returns read text from a directory
    """
    captured_text = ""
    captured_images = []
    directory_path = Path(directory)
    for pdf in directory_path.glob('*.pdf'):
        captured_text += extract_text(pdf)
        captured_images.extend(extract_images(pdf))
    return captured_text, captured_images

