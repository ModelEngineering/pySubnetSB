'''Utility functions for the SIRN package.'''
import collections
from functools import wraps
import numpy as np
import time
import pandas as pd # type: ignore
from typing import List, Tuple, Union

IS_TIMEIT = False

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

def isInt(val: str)->bool:
    """Determines if a string is an integer.

    Args:
        val (str): A string.

    Returns:
        bool: True if the string is an integer.
    """
    try:
        int(val)
        return True
    except ValueError:
        return False

Statistics = collections.namedtuple("Statistics", "mean std min max count total") 
def calculateSummaryStatistics(arr: Union[list, np.ndarray, pd.Series])->Statistics:
    """Calculates basic statistics for an array.

    Args:
        arr (np.array): An array.

    Returns:
        dict: A dictionary with basic statistics.
    """
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    min = float(np.min(arr))
    max = float(np.max(arr))
    count = len(arr)
    total = np.sum(arr)
    return Statistics(mean=mean, std=std, min=min, max=max,
                      count=count, total=total)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        if IS_TIMEIT:
            start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if IS_TIMEIT:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper