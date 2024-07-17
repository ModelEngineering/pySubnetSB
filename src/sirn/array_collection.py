'''Describes a set of arrays and their partition constrained permutations.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.

Terminology:
- Array: A collection of numbers.
- ArrayCollection: A collection of arrays.
- Encoding: A single number that represents the array, a homomorphism (and so is not unique).
"""
import sirn.constants as cn # type: ignore
from sirn.encoding import Encoding  # type: ignore

import collections
import itertools
import numpy as np
import scipy  # type: ignore
import scipy.special  # type: ignore
from typing import Dict

SEPARATOR = 1000 # Separates the counts in a single number
VALUE_SEPARATOR = ','  # Separates the values in a string encoding

EncodingResult = collections.namedtuple('EncodingResult', ['index_dct', 'sorted_mat', 'encodings'])
    #  index_dct: dict: key is encoding_str, value is a list of indexes
    #  sorted_mat: np.ndarray: The columns as sorted array
    #  encodings: list-encoding: indexed by position in the collection, value is the encoding for the indexed array


class ArrayCollection(object):

    def __init__(self, collection: np.ndarray)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
            is_weighted (bool): Weight the value of the i-th element in an array by the sum of non-zero
                elements in the i-th position.
        """
        self.collection = collection
        self.narr, self.length = np.shape(collection)
        #
        if (self.length > SEPARATOR):
            raise ValueError("Matrix is too large to classify. Maximum number of rows, columns is 1000.")
        # Outputs
        encoding_result = self.encode()
        self.sorted_mat = encoding_result.sorted_mat # Sorted array associated with each encoding
        self.index_dct = encoding_result.index_dct
        self.encodings = encoding_result.encodings
        self.sorted_encodings = sorted(self.encodings, key=lambda x: x.encoding_val)
        self.num_partition = len(set([str(e) for e in self.encodings]))  # Number of partitions 

    def __repr__(self)->str:
        return str(self.encodings)

    @property 
    def log_estimate(self)->float:
        """
        Estimates the number of permutations of the ArrayCollection in log10 units. Uses the
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
        lengths = [len(v) for v in self.index_dct.values()]
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
        encodings:list = []
        for idx, arr in enumerate(self.collection):
            encoding = Encoding(arr)
            encoding_val = encoding.encoding_val
            if not encoding_val in index_dct.keys():
                index_dct[encoding_val] = []
                array_dct[encoding_val] = []
            index_dct[encoding_val].append(idx)
            array_dct[encoding_val].append(encoding.sorted_arr)
            encodings.append(encoding)
        # Construct the sorted array
        encoding_strs = list(index_dct.keys())
        encoding_strs.sort()
        lst = []
        for encoding in encoding_strs:
            lst.extend(array_dct[encoding])
        sorted_mat = np.array(lst)
        #
        return EncodingResult(index_dct=index_dct, sorted_mat=sorted_mat, encodings=encodings)
    
    def constrainedPermutationIterator(self, max_num_perm:int=cn.MAX_NUM_PERM):
        """
        Iterates through all permutations of arrays in the ArrayCollection
        that are constrained by the encoding of the arrays.

        returns:
            np.array-int: A permutation of the arrays.
        """
        iter_dct = {e: None for e in self.index_dct.keys()}  # Iterators for each partition
        permutation_dct = {e: None for e in self.index_dct.keys()}  # Iterators for each partition
        idx = 0  # Index of the partition processed in an iteration
        partitions = list(self.index_dct.keys())  # Encodings for the partitions
        partitions.sort() # Sort the partitions by the encoding representation
        perm_count = 0  # Number of permutations returned
        perm_upper_bound = np.prod([scipy.special.factorial(len(v)) for v in self.index_dct.values()])
        # Limit the number of permutations
        perm_upper_bound = min(perm_upper_bound, max_num_perm)
        # Return the permutations
        while perm_count < perm_upper_bound:
            encoding_str = str(partitions[idx])
            # Try to get the next permutation for the current partition
            if permutation_dct[encoding_str] is not None:
                permutation_dct[encoding_str] = next(iter_dct[encoding_str], None)  # type: ignore
            # Handle the case of a missing or exhaustive iterator
            if permutation_dct[encoding_str] is None:
                # Get a valid iterator and permutation for this partition
                iter_dct[encoding_str] = itertools.permutations(self.index_dct[encoding_str])  # type: ignore
                permutation_dct[encoding_str] = next(iter_dct[encoding_str])  # type: ignore
                if idx < self.num_partition - 1:
                    idx += 1  # Move to the next partition to get a valid iterator
                    continue
            # Have a valid iterator and permutations for all partitions
            # Construct the permutation array by flattening the partition_permutations
            idx = 0  # Reset the partition index
            permutation_idxs:list = []
            for encoding in partitions:
                permutation_idxs.extend(permutation_dct[str(encoding)])  # type: ignore
            permutation_arr = np.array(permutation_idxs)
            perm_count += 1
            yield permutation_arr
    
    def subsetIterator(self, other):
        """
        Iterates iterates over subsets of the other ArrayCollection that are compatible with the current ArrayCollection.

        Args:
            other: ArrayCollection

        Yields:
            np.ndarray: indices of other that are compatible with the current ArrayCollection.
        """
        # Find the sets of compatible arrays
         # For each index, the list of indices in other that are compatible with the corresponding index in self.
        collection_candidates = [ [] for _ in self.collection]
        for self_idx, _ in enumerate(self.collection):
            for other_idx in range(len(other.collection)):
                if self.encodings[self_idx].isCompatibleSubset(other.encodings[other_idx]):
                    collection_candidates[self_idx].append(other_idx)
        # Get the subsets
        iter = itertools.product(*collection_candidates)
        for subset in iter:
            # Use this array if element of other is present at most once
            if len(set(subset)) == len(subset):
                yield np.array(subset)