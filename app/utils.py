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

# Specify the local nltk_data directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Check if 'punkt' is already downloaded, if not, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

def get_env_var(var_name):
    """
    Return the queried dotenv variable
    """
    return os.getenv(var_name)

def tokenize_sentence(text):
    """
    Return the tokenized sentence
    """
    return sent_tokenize(text)
