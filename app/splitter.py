"""
This module contains the text splitting and chunking logic used by 
the LLM Application.

Functions:
    intelligent_chunk: Uses NLP to split the text into dynamic chunk sizes
    llm_embedding: Generates the GPT-3 LLM and embedding

References:
    - Natural Language Toolkit Documentation: https://www.nltk.org
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - GeeksforGeeks: https://www.geeksforgeeks.org/tokenize-text-using-nltk-python/
"""

from .utils import tokenize_sentence

def intelligent_chunk(text, max_chunk_size):
    """
    Return the text chunks split using the NLP tokenizer.
    """
    sentences = tokenize_sentence(text)
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk).split()) > max_chunk_size:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

