"""
Discrete Exterior Calculus Module
==================================
"""

import os, pickle

def get_data_dir():
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, 'data')

def get_data(name):
    """
    Load data samples provided by file `name` in dec/data 
    """
    filename = os.path.join(get_data_dir(), name)
    with open(filename, 'r') as f: 
        return pickle.load(f)

def store_data(name, obj):
    """
    Store `obj` in a file `name` in dec/data
    """
    filename = os.path.join(get_data_dir(), name)
    with open(filename, 'w') as f:
        pickle.dump(obj, f)
