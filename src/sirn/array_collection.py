'''Describes a set of arrays and their partition constrained permutations.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.

Terminology:
- Array: A collection of numbers.
- ArrayCollection: A collection of arrays.
- Encoding: A single number that represents the array, a homomorphism (and so is not unique).
"""
import sirn.constants as cn # type: ignore

import collections
import itertools
import numpy as np
import scipy  # type: ignore
import scipy.special  # type: ignore
from typing import Dict

SEPARATOR = 1000 # Separates the counts in a single number
VALUE_SEPARATOR = ','  # Separates the values in a string encoding

EncodingResult = collections.namedtuple('EncodingResult', ['index_dct', 'sorted_mat'])
    #  index_dct: dict: key is a string representation of the sorted array, value is a list of indexes
    #  sorted_mat: np.ndarray: The columns as sorted array


class ArrayCollection(object):

    def __init__(self, collection: np.ndarray, is_weighted:bool=True)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
            is_weighted (bool): Weight the value of the i-th element in an array by the sum of non-zero
                elements in the i-th position.
        """
        self.collection = collection
        self.narr, self.length = np.shape(collection)
        self.is_weighted = is_weighted
        #
        if (self.length > SEPARATOR):
            raise ValueError("Matrix is too large to classify. Maximum number of rows, columns is 1000.")
        # Outputs
        encoding_result = self.encode()
        self.sorted_mat = encoding_result.sorted_mat # Sorted array associated with each encoding
        self.encoding_dct = encoding_result.index_dct
        encodings = list(self.encoding_dct.keys())
        encodings.sort()
        self.encoding_arr = np.array(encodings)   # Encodings of the arrays
        self.num_partition = len(self.encoding_arr)
        self.log_estimate = self.logEstimateNumPermutations()  # log10 of the estimated number of permutations

    def __repr__(self)->str:
        return str(self.encoding_arr)
    
    def logEstimateNumPermutations(self)->float:
        """
        Estimates the number of permutations of the ArrayCollection in log units. Uses the
        continuous approximation, which is more accurate for large numbers of permutations.

        Returns:
            float: The estimated number of permutations.
        """
        def factorial(n):
            if n == 0:
                return 1
            if n < 15:
                return np.log10(scipy.special.factorial(n))
            else:
                return n*np.log10(n) - n
        #
        lengths = [len(v) for v in self.encoding_dct.values()]
        log_estimate = np.sum([factorial(v) for v in lengths])
        return np.sum(log_estimate)
    
    def isCompatible(self, other)->bool:
        """
        Checks if the two ArrayCollection have the same encoding.

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if not np.allclose(self.sorted_mat.shape, other.sorted_mat.shape):
            return False
        return np.allclose(self.sorted_mat, other.sorted_mat)
    
    # This method can be overridden to provide alternative encodings
    def encode(self)->EncodingResult:
        """Constructs an encoding for an ArrayCollection. The encoding is
        a string representation of the sorted array. The encoding is the value
        separated by ".".

        Args:
            arr (np.ndarray): _description_

        Returns: EncodingResult
            index_dct: dict: key is a string representation of the sorted array, value is a list of indexes
            array_dct np.ndarray: The sorted array
            
        """
        index_dct: dict = {}
        array_dct: dict = {}
        for idx, matrix in enumerate(self.collection):
            sorted_arr = np.sort(matrix)
            this_encoding = str(sorted_arr)
            this_encoding = this_encoding.replace('[', '')
            this_encoding = this_encoding.replace(']', '')
            this_encoding = this_encoding.replace(' ', VALUE_SEPARATOR)
            if not this_encoding in index_dct.keys():
                index_dct[this_encoding] = []
                array_dct[this_encoding] = []
            index_dct[this_encoding].append(idx)
            array_dct[this_encoding].append(sorted_arr)
        # Construct the sorted array
        encodings = list(index_dct.keys())
        encodings.sort()
        lst = []
        for encoding in encodings:
            lst.extend(array_dct[encoding])
        sorted_mat = np.array(lst)
        #
        return EncodingResult(index_dct=index_dct, sorted_mat=sorted_mat)
    
    def partitionPermutationIterator(self, max_num_perm:int=cn.MAX_NUM_PERM):
        """
        Iterates through all permutations of arrays in the ArrayCollection
        that are constrained by the encoding of the arrays.

        returns:
            np.array-int: A permutation of the arrays.
        """
        iter_dct = {e: None for e in self.encoding_dct.keys()}  # Iterators for each partition
        permutation_dct = {e: None for e in self.encoding_dct.keys()}  # Iterators for each partition
        idx = 0  # Index of the partition processed in an iteration
        count = 0
        max_count = np.prod([scipy.special.factorial(len(v)) for v in self.encoding_dct.values()])
        max_count = min(max_count, max_num_perm)
        while count < max_count:
            cur_encoding = self.encoding_arr[idx]
            # Try to get the next permutation for the current partition
            if permutation_dct[cur_encoding] is not None:
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding], None)  # type: ignore
            # Handle the case of a missing or exhaustive iterator
            if permutation_dct[cur_encoding] is None:
                # Get a valid iterator and permutation for this partition
                iter_dct[cur_encoding] = itertools.permutations(self.encoding_dct[cur_encoding])  # type: ignore
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding])  # type: ignore
                if idx < len(self.encoding_arr)-1:
                    idx += 1  # Move to the next partition to get a valid iterator
                    continue
            # Have a valid iterator and permutations for all partitions
            # Construct the permutation array by flattening the partition_permutations
            idx = 0  # Reset the partition index
            permutation_idxs:list = []
            for encoding in self.encoding_arr:
                permutation_idxs.extend(permutation_dct[encoding])  # type: ignore
            permutation_arr = np.array(permutation_idxs)
            count += 1
            yield permutation_arr