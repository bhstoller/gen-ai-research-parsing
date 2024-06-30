"""
This module contains the text splitting logic.

It contains functions to split the text.
"""

from langchain.text_splitter import CharacterTextSplitter

def split_text(raw_text, seperator= "\n", chunk_size= 800, chunk_overlap= 200):
    """
    Return a text splitter
    """
    splitter = CharacterTextSplitter(
        separator= seperator,
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap
    )
    splitted_text = splitter.split_text(raw_text)
    return splitted_text

# def split_text(raw_text, splitter):
#     """
#     Return the split text
#     """
#     texts= splitter.split_text(raw_text)
#     return texts
