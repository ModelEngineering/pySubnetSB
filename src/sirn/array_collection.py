'''Describes a set of arrays and their partition constrained permutations.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.

Terminology:
- Array: A collection of numbers.
- ArrayCollection: A collection of arrays.
- Encoding: A single number that represents the array, a homomorphism (and so is not unique).
"""

import itertools
import numpy as np
import scipy  # type: ignore
from typing import Dict

SEPARATOR = 1000 # Separates the counts in a single numbera


class ArrayCollection(object):

    def __init__(self, collection: np.ndarray)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
        """
        self.collection = collection
        self.narr, self.length = np.shape(collection)
        #
        if (self.length > SEPARATOR):
            raise ValueError("Matrix is too large to classify. Maximum number of rows, columns is 1000.")
        # Outputs
        self.encoding_dct = self.encode()
        encodings = list(self.encoding_dct.keys())
        encodings.sort()
        self.encoding_arr = np.array(encodings)   # Encodings of the arrays
        self.num_partition = len(self.encoding_arr)

    def __repr__(self)->str:
        return str(self.encoding_arr)
    
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
    def encode(self)->Dict[int, np.ndarray]:
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
    
    def partitionPermutationIterator(self):
        """
        Iterates through all permutations of arrays in the ArrayCollection
        that are constrained by the encoding of the arrays.

        returns:
            np.array-int: A permutation of the arrays.
        """
        iter_dct = {e: None for e, v in self.encoding_dct.items()}  # Iterators for each partition
        permutation_dct = {e: None for e, v in self.encoding_dct.items()}  # Iterators for each partition
        idx = 0  # Index of the partition processed in an iteration
        max_count = np.prod([scipy.special.factorial(len(v)) for v in self.encoding_dct.values()])
        count = 0
        while count < max_count:
            cur_encoding = self.encoding_arr[idx]
            # Try to get the next permutation for the current partition
            if permutation_dct[cur_encoding] is not None:
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding], None)
            # Handle the case of a missing or exhaustive iterator
            if permutation_dct[cur_encoding] is None:
                # Get a valid iterator and permutation for this partition
                iter_dct[cur_encoding] = itertools.permutations(self.encoding_dct[cur_encoding])
                permutation_dct[cur_encoding] = next(iter_dct[cur_encoding])
                if idx < len(self.encoding_arr)-1:
                    idx += 1  # Move to the next partition to get a valid iterator
                    continue
            # Have a valid iterator and permutations for all partitions
            # Construct the permutation array by flattening the partition_permutations
            idx = 0  # Reset the partition index
            permutation_idxs = []
            for encoding in self.encoding_arr:
                permutation_idxs.extend(permutation_dct[encoding])
            permutation_arr = np.array(permutation_idxs)
            count += 1
            yield permutation_arr