"""
This module contains the arithmetic text analysis logic used by
the LLM Application.

Functions:
    analyze_length: Computes the mean, standard deviation, and maximum of each document

References:
    - OpenAI GPT-3 (ChatGPT): https://chatgpt.com
"""

from pathlib import Path
import numpy as np
from .reader import extract_text

def analyze_length(directory):
    """
    Return the mean, standard deviation, and maximum length of each
    PDF document in the directory.
    """
    lengths = []
    for pdf in Path(directory).glob('*.pdf'):
        try:
            text = extract_text(pdf)
            lengths.append(len(text.split()))
        except Exception as e:
            print(f"Could not read {pdf}: {e}")
    return np.mean(lengths), np.std(lengths), np.max(lengths)
