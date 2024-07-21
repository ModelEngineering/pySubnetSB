'''Describes a set of arrays and provides an iterator over partition constrained permutations.'''

"""
Encodings of arrays are central to the speedup provided by DSIRN to avoid making array comparisons.
An encoding has one or more encoding criteria. These are order independent boolean values that describe
an array such as the number of 0 values. An encoding vector for an array is the count of each array condition
that is satisfied, which is independent of the length of the array. An encoding for an array collection
is an encoding vector for each array in the collection.

The encoding criteria used here are the ranges of array values with respect to a set of boundary values.

To do
1. Add encoding for pairs of arrays.
2. Add isSubsetCompatible method that uses row pairs.
3. Integrate into ArrayCollection iterators.
"""

import collections
import numpy as np
from typing import List, Tuple

SEPARATOR = 1000 # Separates the counts in a single number
VALUE_SEPARATOR = ','  # Separates the values in a string encoding
BOUNDARY_VALUES = [-1, 0, 1]
 # Values that are counted are in the ranges [-inf, -1), [-1, -1], (-1, 0), [0, 0], (0, 1), [1, 1], [1, inf)]
ENCODING_BASE = 1000

EncodingResult = collections.namedtuple('EncodingResult', ['index_dct', 'sorted_mat', 'encodings'])
    #  index_dct: dict: key is encoding_str, value is a list of indexes
    #  sorted_mat: np.ndarray: The columns as sorted array
    #  encodings: list-encoding: indexed by position in the collection, value is the encoding for the indexed array


class Encoding(object):

    def __init__(self, collection: np.ndarray)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
            is_weighted (bool): Weight the value of the i-th element in an array by the sum of non-zero
                elements in the i-th position.
        """
        self.collection = collection
        self.num_row, self.num_column = np.shape(collection)
        #
        # Outputs
        #  encoding_mat: matrix of encodings (column) for each array in the collection (row).
        #  encodings: list of encodings for each array in the collection.
        self.encoding_mat, self.encodings = self._makeEncodingMat()
        # Sorted list of unique encodings
        self.unique_encodings = list(set(self.encodings))
        self.unique_encodings.sort()
        # Indices of arrays with the same encoding
        self.index_dct = {self.encodings[n]: n for n in range(len(self.encodings))}
        # Construct the dictionary of encodings
        self.encoding_dct:dict = {k: [] for k in set(self.encodings)}
        # encoding_dct: key is encoding, value is a list of indexes with that encoding
        {self.encoding_dct[self.encodings[i]].append(i) for i in range(len(self.encodings))}
        #
        self.num_partition = len(self.encoding_dct)

    def __repr__(self)->str:
        return str(self.encoding_mat)
    
    def _makeEncodingMat(self)->Tuple[np.ndarray, List[int]]:
        """
        Constructs the encoding matrix.
        Columns are the counts
          0: [-inf, -1) 
          1: [-1, -1] 
          2: (-1, 0) 
          3: [0, 0]
          4: (0, 1)
          5: [1, 1]
          6: (1, inf)]
        Rows correspond to arrays in the collection

        Returns:
            np.ndarray: The encoding matrix.
        """
        encoding_mat = np.zeros((self.num_row, 2*len(BOUNDARY_VALUES) + 1))
        for irow in range(self.num_row):
            encoding_mat[irow, 0] = np.sum([ x < BOUNDARY_VALUES[0] for x in self.collection[irow, :]])
            encoding_mat[irow, 2] = np.sum([ (x > BOUNDARY_VALUES[0]) and (x < BOUNDARY_VALUES[1])
                                            for x in self.collection[irow, :]])
            encoding_mat[irow, 6] = np.sum([x  > BOUNDARY_VALUES[2] for x in self.collection[irow, :]])
            for icol, value in zip([1, 3, 5], BOUNDARY_VALUES):
                encoding_mat[irow, icol] = np.sum(self.collection[irow, :] == value)
        # Construct the dictionary of encodings
        encodings:List[int] = []
        num_column = encoding_mat.shape[1]
        multiplers = np.array([ENCODING_BASE**n for n in range(num_column)])
        for arr in encoding_mat:
            if np.any(arr > ENCODING_BASE):
                import pdb; pdb.set_trace()
                raise ValueError("Encoding value is too large")
            encodings.append(int(np.sum(arr*multiplers)))
        return encoding_mat, encodings
    
    def __eq__(self, other)->bool:
        """
        Checks if the two ArrayCollectionEncoding have the same encoding.

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if np.allclose(self.encoding_mat.shape, other.encoding_mat.shape):
            return np.allclose(self.encoding_mat, other.encoding_mat)
        return False

    def isSubsetCompatible(self, other)->bool:
        """
        Determines if the ArrayCollectionEncoding is a subset compatible with other.

        Args:
            other (ArrayCollectionEncoding): An ArrayCollectionEncoding.

        Returns:
            bool: True if the ArrayCollectionEncoding is a subset of the other ArrayCollectionEncoding.
        """
        for idx1 in range(len(self.collection)):
            is_found = False
            for idx2 in range(len(other.collection)):
                if np.all(self.collection[idx1] <= other.collection[idx2]):
                    is_found = True
                    break
            if not is_found:
                return False
        return True