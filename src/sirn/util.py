'''Utility functions for the SIRN package.'''
import collections
from functools import wraps, cmp_to_key
import numpy as np
import time
import pandas as pd # type: ignore
from typing import List, Tuple, Union

IS_TIMEIT = False
ArrayContext = collections.namedtuple('ArrayContext', "string, num_row, num_column")

#def string2Array(array_str: str)->np.ndarray:
#    """Converts a string to an array.
#
#    Args:
#        array_str (str): An string constructed by str(np.array).
#
#    Returns:
#        np.array: An array.
#    """
#    array_str = array_str.replace('\n', ',')
#    array_str = array_str.replace(' ', ', ')
#    array_str = array_str.replace('\n', '')
#    while True:
#        if ",," not in array_str:
#            break
#        array_str = array_str.replace(",,", ",")
#    return np.array(eval(array_str))

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

def deprecatedmakeRowOrderIndependentHash(array:np.ndarray)->int:
    """Creates a single integer hash for a 1 or 2, or 3 dimensional array
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
    
def makeRowOrderIndependentHash(array:np.ndarray)->int:
    """Creates a single integer hash for a 1 or 2 dimensional array
    that depends only on the order of values in the columns (last dimension in the array).
    So, the resulting hash is invariant to permutations of the rows.

    Args:
        array (np.array): An array.

    Returns:
        int: Hash value.
    """
    #####
    def hashRow(row):
        return np.sum(pd.util.hash_array(row))
    #####
    def hash2DArray(array):
        result = []
        for row in array:
            result.append(hashRow(row))
        return np.sum(result)
    #
    if array.ndim == 1:
        return hashRow(array)
    elif array.ndim == 2:
        return hash2DArray(array)
    else:
        raise ValueError("Array must be 1, 2 dimensional.")
    
def isArrayLessEqual(left_arr:np.ndarray, right_arr:np.ndarray)->bool:
    """Determines if one array is less than another.

    Args:
        left_arr (np.array): An array.
        right_arr (np.array): An array.

    Returns:
        bool: True if left_arr is less than right_arr.
    """
    if left_arr.shape != right_arr.shape:
        return False
    for left_val, right_val in zip(left_arr, right_arr):
        if left_val < right_val:
            return True
        elif left_val > right_val:
            return False
    return True

def arrayToStr(arr:np.ndarray)->str:
    """Converts an array of integers to a single integer.

    Args:
        arr (np.array): An array of integers.

    Returns:
        int: The integer value of the array.
    """
    return ''.join(map(str, arr))

def arrayToSortedDataFrame(array:np.ndarray)->pd.DataFrame:
    """Converts an array to a sorted pandas DataFrame.

    Args:
        array (np.array): A 2d array.

    Returns:
        pd.DataFrame: A sorted DataFrame.
    """
    sorted_assignment_arr = sorted(array, key=arrayToStr)
    return pd.DataFrame(sorted_assignment_arr)

def pruneArray(array:np.ndarray, max_size:int)->Tuple[np.ndarray, bool]:
    """
    Randomly prunes an array to a maximum size.

    Args:
        array (np.array): A 2d array.
        max_size (int): The maximum number of rows to keep.

    Returns:
        np.array: A pruned array.
    """
    if array.shape[0] <= max_size:
        return array, False
    idxs = np.random.permutation(array.shape[0])[:max_size]
    return array[idxs], True

def array2Context(array:np.ndarray)->ArrayContext:
    array = np.array(array)
    if array.ndim == 1:
        num_column = len(array)
        num_row = 1
    elif array.ndim == 2:
        num_row, num_column = np.shape(array)
    else:
        raise ValueError("Array must be 1 or 2 dimensional.")
    flat_array = np.reshape(array, num_row*num_column)
    str_arr = [str(i) for i in flat_array]
    array_str = "[" + ",".join(str_arr) + "]"
    return ArrayContext(array_str, num_row, num_column)

def string2Array(array_context:ArrayContext)->np.ndarray:
    array = np.array(eval(array_context.string))
    array = np.reshape(array, (array_context.num_row, array_context.num_column))
    return array