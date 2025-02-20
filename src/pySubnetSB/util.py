'''Utility functions for the SIRN package.'''
import collections
from functools import wraps, cmp_to_key
import itertools
import json
import numpy as np
import tempfile
import time
import pandas as pd # type: ignore
from typing import List, Tuple, Union, Optional
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings from urllib3
import urllib3  # type: ignore

IS_TIMEIT = False
ArrayContext = collections.namedtuple('ArrayContext', "string, num_row, num_column")
INT_MAX = 1000000

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
    
def deprecatedhashMatrix(matrix:np.ndarray)->int:
    """Creates a single integer hash for a 2 dimensional array.

    Args:
        array (np.array): An 2d array.

    Returns:
        int: Hash value.
    """
    # Encode rows
    is_0_arr = matrix == 0
    is_1_arr = matrix == 1
    is_minus_1_arr = matrix == -1
    is_not_arr = np.logical_or(is_0_arr, is_1_arr)
    is_not_arr = np.logical_or(is_minus_1_arr, is_not_arr)
    is_not_arr = np.logical_not(is_not_arr)
    values = np.sum(is_0_arr, axis=1)
    values += 1000*np.sum(is_1_arr, axis=1)
    values += 1000000*np.sum(is_minus_1_arr, axis=1)
    values += 1000000000*np.sum(is_not_arr, axis=1)
    result = hash(str(pd.util.hash_array(np.sort(values))))
    return result

def hashMatrix(matrix:np.ndarray)->np.int64:
    """Creates a single integer hash for a 2 dimensional array. Matrix cannot have a dimensin > 100.

    Args:
        array (np.array): An 2d array.

    Returns:
        int64: Hash value.
    """
    for dim in matrix.shape:
        if dim > 100:
            raise ValueError("Matrix cannot have a dimension > 100.")
    #####
    def hashRows(array:np.ndarray)->np.int64:
        VALUES = [-2, -1, 0, 1, 2]
        results = []
        for row in array:
            row_encoding = 0
            for idx, val in enumerate(VALUES):
                row_encoding += (10**idx)*np.sum(row == val)
            results.append(row_encoding)
        result_arr = np.sort(np.array(results, dtype=np.int64))
        result = np.int64(np.sum(pd.util.hash_array(result_arr)))
        return result
    #####
    row_hash = hashRows(matrix)
    column_hash = hashRows(matrix.T)
    # Hanlde possible overflow on summation
    try:
        np.seterr(over='raise')
        final_hash = row_hash + column_hash
    except:
        final_hash = np.int64(0.1*row_hash + 0.1*column_hash)
    return final_hash
    
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

def sampleListOfLists(list_of_lists:List[List[int]], num_samples:int)->np.ndarray:
    """Randomly samples the permutations implied by a list of lists.

    Args:
        list_of_lists (List[List[int]]): A list of lists.
        num_samples (int): Number of samples.

    Returns:
        np.ndarray: An array of samples. Columns are instances from list, rows are samples.
    """
    lengths = [len(lst) for lst in list_of_lists]
    arrays = [np.array(lst) for lst in list_of_lists]
    sample_position_arr = np.array([np.random.randint(0, l, num_samples) for l in lengths]).T
    samples = []
    for sample_position in sample_position_arr:
        sample = [a[s] for a, s in zip(arrays, sample_position)]
        samples.append(sample)
    return np.array(samples)

def partitionArray(array:np.ndarray, num_partition:int)->List[np.ndarray]:
    """Partitions an array into num_partitions.

    Args:
        array (np.array): An array.
        num_partitions (int): Number of partitions.

    Returns:
        List[np.ndarray]: A list of partitions.
    """
    actual_num_partition = min(num_partition, len(array))
    partitions:list = [ [] for lst in range(actual_num_partition)]
    [partitions[n%actual_num_partition].append(array[n].tolist()) for n in range(len(array))]
    partitions = [np.array(partition) for partition in partitions]
    return partitions

def serializeDct(dct:dict)->str:
    """Serializes a dictionary.

    Args:
        dct (dict): A dictionary.

    Returns:
        str: A serialized dictionary.
    """
    for key, val in dct.items():
        if isinstance(val, np.ndarray):
            dct[key] = val.tolist()
    return json.dumps(dct)

def selectRandom(array:np.ndarray, num_select:int)->np.ndarray:
    """Randomly selects elements from an array.

    Args:
        array (np.array): An array.
        num_select (int): Number of elements to select.

    Returns:
        np.array: An array of selected elements.
    """
    idxs = np.random.permutation(len(array))[:num_select]
    return array[idxs]

def getDefaultSpeciesNames(num_species:int)->np.ndarray:
    """Creates default names for species.

    Args:
        num_species (int)

    Returns:
        np.ndarray[str]: species names
    """
    return np.array([f"S{i}" for i in range(num_species)])


def getDefaultReactionNames(num_reaction:int)->np.ndarray:
    """Creates default names for reaction.

    Args:
        num_reaction (int)

    Returns:
        np.ndarray[str]: reaction names
    """
    return np.array([f"J{i}" for i in range(num_reaction)])

def encodeIntPair(int1:Union[int, np.ndarray], int2:Union[int, np.ndarray],
      max_int:int=INT_MAX)->Union[np.int64, np.ndarray]:
    """Encodes a pair of integers as a single integer.

    Args:
        int1 (int | p.ndarray): An integer.
        int2 (int | np.ndarray): An integer.
        max_int (int): The maximum integer value. Defaults to 1e6

    Returns:
        int | np.ndarray: An encoded integer.
    """
    if isinstance(int1, np.ndarray):
        is_int = False
        arr1:np.ndarray = int1.astype(np.int64)
        arr2:np.ndarray = int2.astype(np.int64)  # type: ignore
    else:
        is_int = True
        arr1 = np.array([int1]).astype(np.int64)
        arr2 = np.array([int2]).astype(np.int64)
    if any(arr1 > max_int) or any(arr2 > max_int):
        raise ValueError(f"Integers must be less than {max_int}.")
    result = arr1*max_int + arr2
    if is_int:
        return result[0]
    else:
        return result

def decodeIntPair(encoded_int:Union[np.int64, np.ndarray],
      max_int:int=INT_MAX)->Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """Encodes a pair of integers as a single integer.

    Args:
        encoded_int(int64 | np.ndarray): An integer.
        max_int (int): The maximum integer value. Defaults to 1e6

    Returns:
        int: An encoded integer.
    """
    int2 = encoded_int % max_int
    if isinstance(int2, np.ndarray):
        int2 = [int(i) for i in int2] # type: ignore
    else:
        int2 = int(int2)  # type: ignore
    int1 = (encoded_int - int2)/max_int
    if isinstance(int1, np.ndarray):
        int1 = [int(i) for i in int1] # type: ignore
    else:
        int1 = int(int1)  # type: ignore
    return int1, int2  # type: ignore

def getAllSubsets(a_list)->list:
    """Generates all subsets of a list.

    Args:
        a_list (list): A list.

    Returns:
        list: A list of all subsets.
    """
    result:list = [[]]
    if len(a_list) == 0:
        return result
    #
    for size in range(1, len(a_list)+1):
        for subset in itertools.combinations(a_list, size):
            result.append(list(subset))
    # 
    return result

def getStringsFromURL(url:str)->List[str]:
    """Downloads a file from a URL and convert to list of strings.

    Args:
        url (str): URL of the file.

    Returns:
        List[str]: List of strings.
    """
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    if resp.status != 200:
        raise ValueError(f"Failed to download file from {url}")
    return resp.data.decode("utf-8").split("\n")

def makeLocalFileFromURL(url:str, local_file:Optional[str]=None)->str:
    """Downloads a file from a URL and saves it locally.

    Args:
        url (str): URL of the file.
        local_file (str): Local file name.

    Returns:
        str: Local file name.
    """
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    if resp.status != 200:
        raise ValueError(f"Failed to download file from {url}")
    strings = getStringsFromURL(url)
    if local_file is None:
        fp = tempfile.NamedTemporaryFile(delete=False)
        bytes = '\n'.join(strings).encode("utf-8")
        fp.write(bytes)
        local_file = fp.name
        fp.close()
    else:
        with open(local_file, "w") as fp:  # type: ignore
            fp.write(resp.data)
    return local_file