"""
This module contains the utility variables needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(var_name):
    """
    Return the quieried dotenv variable
    """
    return os.getenv(var_name)
