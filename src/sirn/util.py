'''Utility functions for the SIRN package.'''
import collections
from functools import wraps
import numpy as np
import time
import pandas as pd # type: ignore
from typing import List, Tuple, Union

IS_TIMEIT = False

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

def repeatArray(array:np.ndarray, num_repeat:int)->np.ndarray:
    """Creates a two dimensional array consisting of num_repeat blocks of the input array.

    Args:
        array (np.array): An array.
        num_repeat (int): Number of times to repeat the array.

    Returns:
        np.array: array.columns X array.rows*num_repeats
    """
    return np.vstack([array]*num_repeat)

def repeatRow(array:np.ndarray, num_repeat:int)->np.ndarray:
    """Creates a two dimensional array consisting of num_repeat repetitions of each
    row of the input array.

    Args:
        array (np.array): An array.
        num_repeats (int): Number of times to repeat the array.

    Returns:
        np.array: array.columns X array.rows*num_repeats
    """
    repeat_arr = np.repeat(array, num_repeat, axis=0)
    return repeat_arr

def arrayProd(arr:np.ndarray)->float:
    """Calculates the product of the elements of an array in log space
    to avoid overflow.

    Args:
        arr (np.array): An array.

    Returns:
        int: The product of the elements of the array.
    """
    return np.exp(np.sum(np.log(arr)))

def makeRowOrderIndependentHash(array:np.ndarray)->int:
    """Creates a single integer hash for a 1, 2, or 3 dimensional array
    that depends only on the order of values in the columns (last dimension in the array).
    So, the resulting hash is invariant to permutations of the rows.

    Args:
        array (np.array): An array.

    Returns:
        int: Hash value.
    """
    def hash2DArray(array):
        result = []
        for row in array:
            result.append(hash(str(row)))
        return hash(str(np.sort(np.array(result))))
    #
    if array.ndim == 1:
        return hash(str(array))
    elif array.ndim == 2:
        return hash2DArray(array)
    elif array.ndim == 3:
        result = []
        for mat in array:
            result.append(hash2DArray(mat))
        return hash(str(np.sort(np.array(result))))
    else:
        raise ValueError("Array must be 1, 2, or 3 dimensional.")