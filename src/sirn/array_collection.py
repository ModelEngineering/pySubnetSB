'''Describes a set of arrays and their partition constrained permutations.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.

Terminology:
- Array: A collection of numbers.
- ArrayCollection: A collection of arrays.
- Encoding: A single number that represents the array, a homomorphism (and so is not unique).
"""
import sirn.constants as cn # type: ignore
from src.sirn.encoding import Encoding  # type: ignore

import collections
import itertools
import numpy as np
import scipy  # type: ignore
import scipy.special  # type: ignore

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
        self.encoding = Encoding(collection)

    def __repr__(self)->str:
        return str(self.encoding.encodings)

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
        lengths = [len(v) for v in self.encoding.encoding_dct.values()]
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
        this_arr = np.array(self.encoding.encodings)
        this_arr = np.sort(this_arr)
        other_arr = np.array(other.encoding.encodings)
        other_arr = np.sort(this_arr)
        if len(this_arr) != len(other_arr):
            return False
        return np.allclose(this_arr, other_arr)

    @staticmethod
    def _findCompatibleRows(subset_mat:np.array, superset_mat:np.array)->list:
        """
        Finds for each row in subset, the collection of rows in subset such the row in subset <= row in superset.

        Args:
            subset_mat: np.ndarray (subset_num_row, num_col)
            superset_mat: np.ndarray (superset_num_row, num_col)

        Returns:
            List[np.ndarry]: list of the rows in subset <= row in superset
        """
        num_row_subset, _ = np.shape(subset_mat)
        num_row_superset, _ = np.shape(superset_mat)
        # Expand the matrices
        xsubset_mat:np.ndarray = np.repeat(subset_mat, num_row_superset, axis=0)
        xsuperset_mat:np.ndarray = np.concatenate([superset_mat for _ in range(num_row_subset)])
        # Do the comparisons. Rows are the rows in superset; columns are rows in subset.
        # We want the indices of rows in superset for each column (row in subset)
        comparison_arr = np.all(xsubset_mat <= xsuperset_mat, axis=1)
        index_arr = np.array(range(num_row_superset))  # All superset rows
        # For each row n in subset (column in comparison_arr), select those rows in superset
        # that satisfy the inequality.
        compatibles = [index_arr[comparison_arr[n*num_row_superset:(n+1)*num_row_superset]] for n in range(num_row_subset)]
        return compatibles
     
    def constrainedPermutationIterator(self, max_num_perm:int=cn.MAX_NUM_PERM):
        """
        Iterates through all permutations of arrays in the ArrayCollection
        that are constrained by the encoding of the arrays.

        returns:
            np.array-int: A permutation of the arrays.
        """
        iter_dct = {e: None for e in self.encoding.unique_encodings}  # Iterators for each partition
        permutation_dct = {e: None for e in self.encoding.unique_encodings}  # Iterators for each partition
        idx = 0  # Index of the partition processed in an iteration
        partitions = list(self.encoding.unique_encodings)  # Encodings for the partitions
        perm_count = 0  # Number of permutations returned
        perm_upper_bound = np.prod([scipy.special.factorial(len(v))
                for v in self.encoding.encoding_dct.values()])
        # Limit the number of permutations
        perm_upper_bound = min(perm_upper_bound, max_num_perm)
        # Return the permutations
        while perm_count < perm_upper_bound:
            cur_encoding = partitions[idx]
            # Try to get the next permutation for the current partition
            if permutation_dct[cur_encoding] is not None:
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding], None)  # type: ignore
            # Handle the case of a missing or exhaustive iterator
            if permutation_dct[cur_encoding] is None:
                # Get a valid iterator and permutation for this partition
                iter_dct[cur_encoding] = itertools.permutations(self.encoding.encoding_dct[cur_encoding])  # type: ignore
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding])  # type: ignore
                if idx < self.encoding.num_partition - 1:
                    idx += 1  # Move to the next partition to get a valid iterator
                    continue
            # Have a valid iterator and permutations for all partitions
            # Construct the permutation array by flattening the partition_permutations
            idx = 0  # Reset the partition index
            permutation_idxs:list = []
            for encoding in partitions:
                permutation_idxs.extend(permutation_dct[encoding])  # type: ignore
            permutation_arr = np.array(permutation_idxs)
            perm_count += 1
            yield permutation_arr
    
    def subsetIterator(self, other):
        """
        Iterates iterates over subsets of arrays in other that are compatible with the current ArrayCollection.

        Args:
            other: ArrayCollection

        Yields:
            np.ndarray: indices of other that are compatible with the current ArrayCollection.
        """
        # Find the sets of compatible arrays
         # For each index, the list of indices in other that are compatible with the corresponding index in self.
        collection_candidates = self._findCompatibleRows(self.encoding.encoding_mat,
                other.encoding.encoding_mat)
        # Get the subsets
        iter = itertools.product(*collection_candidates)
        for subset in iter:
            # Use this array if element of other is present at most once
            if len(set(subset)) == len(subset):
                yield np.array(subset)