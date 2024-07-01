"""
This module contains utility variable logic used by the other modules.

Functions:
    get_env_var: Returns the queried secret variable in .env
    tokenize_sentence: Returns the tokenized sentence

References:
    - Natural Language Toolkit Documentation: https://www.nltk.org
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
    - GeeksforGeeks: https://www.geeksforgeeks.org/tokenize-text-using-nltk-python/
"""

import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

# Load the .env file
load_dotenv()

# Download the NLP model
nltk.download('punkt')

def get_env_var(var_name):
    """
    Return the quieried dotenv variable
    """
    return os.getenv(var_name)

def tokenize_sentence(text):
    """
    Return the tokenized sentence
    """
    return sent_tokenize(text)