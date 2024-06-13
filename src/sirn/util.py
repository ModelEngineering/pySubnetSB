'''Utility functions for the SIRN package.'''
import numpy as np
import pandas as pd # type: ignore

def hashArray(arr: np.ndarray)->int:
    """Hashes an array.

    Args:
        arr (np.array): An array.

    Returns:
        int: Hash value.
    """
    if len(arr) == 0:
        return 0
    hash_vals = pd.util.hash_array(arr)
    hash_val = hash_vals[0]
    if len(hash_vals) == 1:
        return hash_val
    for val in hash_vals[1:]:
        hash_val = pd.util.hash_array(np.array([hash_val ^ val]))[0]
    return hash_val

def string2Array(array_str: str)->np.ndarray:
    """Converts a string to an array.

    Args:
        array_str (str): An string constructed by str(np.array).

    Returns:
        np.array: An array.
    """
    array_str = array_str.replace('\n', ',')
    array_str = array_str.replace(' ', ', ')
    array_str = array_str.replace('\n', '')
    while True:
        if ",," not in array_str:
            break
        array_str = array_str.replace(",,", ",")
    return np.array(eval(array_str))