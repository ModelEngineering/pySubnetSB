'''Describes a set of arrays and their partition constrained permutations.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.

Terminology:
- Array: A collection of numbers.
- ArrayCollection: A collection of arrays.
- Encoding: A single number that represents the array, a homomorphism (and so is not unique).
"""
import sirn.constants as cn

import itertools
import numpy as np
import scipy  # type: ignore
import scipy.special  # type: ignore
from typing import Dict

SEPARATOR = 1000 # Separates the counts in a single numbera


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
        self.encoding_dct = self.encode()
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
        if len(self.encoding_arr) != len(other.encoding_arr):
            return False
        result = np.allclose(self.encoding_arr, other.encoding_arr)
        return result

    # This method can be overridden to provide alternative encodings
    def deprecatedEncode(self)->Dict[int, np.ndarray]:
        """Constructs an encoding for an ArrayCollection.

        Args:
            arr (np.ndarray): _description_

        Returns:
            np.ndarray: key: encoding, values: indexes of arrays with the same encoding
        """
        dct: dict = {}
        for idx, arr in enumerate(self.collection):
            this_encoding = np.sum(arr < 0) + np.sum(arr == 0) * SEPARATOR + np.sum(arr > 0)*SEPARATOR**2
            if not this_encoding in dct.keys():
                dct[this_encoding] = []
            dct[this_encoding].append(idx)
        return dct
    
    # This method can be overridden to provide alternative encodings
    def encode(self)->Dict[int, np.ndarray]:
        """Constructs an encoding for an ArrayCollection.

        Args:
            arr (np.ndarray): _description_

        Returns:
            np.ndarray: key: encoding, values: indexes of arrays with the same encoding
        """
        dct: dict = {}
        if self.is_weighted:
            weight_arr = np.abs(self.collection).sum(axis=0)
        else:
            weight_arr = np.ones(self.length)
        for idx, arr in enumerate(self.collection):
            new_arr = arr*weight_arr
            this_encoding = np.sum(new_arr < 0) + np.sum(new_arr == 0)*SEPARATOR + np.sum(new_arr > 0)*SEPARATOR**2
            if not this_encoding in dct.keys():
                dct[this_encoding] = []
            dct[this_encoding].append(idx)
        return dct
    
    def partitionPermutationIterator(self, max_num_perm:int=cn.MAX_NUM_PERM):
        """
        Iterates through all permutations of arrays in the ArrayCollection
        that are constrained by the encoding of the arrays.

        returns:
            np.array-int: A permutation of the arrays.
        """
        iter_dct = {e: None for e, v in self.encoding_dct.items()}  # Iterators for each partition
        permutation_dct = {e: None for e, v in self.encoding_dct.items()}  # Iterators for each partition
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