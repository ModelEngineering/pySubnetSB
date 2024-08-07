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
from sirn.named_matrix import NamedMatrix  # type: ignore

import collections
import itertools
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Tuple

VALUE_SEPARATOR = ','  # Separates the values in a string encoding
INDEX_SEPARATOR = '_'  # Separates the indexes in a string encoding
BOUNDARY_VALUES = [-1, 0, 1]
 # Values that are counted are in the ranges [-inf, -1), [-1, -1], (-1, 0), [0, 0], (0, 1), [1, 1], [1, inf)]
# Index of the predicate is its encoding
CRITERIA = [
    lambda x: x < -1,                              # 0
    lambda x: x == -1,                             # 1
    lambda x: np.logical_and((x > -1), (x < 0)),   # 2
    lambda x: x == 0,                              # 3
    lambda x: np.logical_and((x > 0), (x < 1)),    # 4
    lambda x: x == 1,                              # 5
    lambda x: x > 1,                               # 6
]
ENCODING_VAL = 1000  # Base of exponent used to separate encoding values, the count of criteria occurrences
CRITERIA_VALUES = np.array(range(len(CRITERIA)))
CRITERIA_PAIRED_VALUES = np.array([i*ENCODING_VAL + j for i in CRITERIA_VALUES for j in CRITERIA_VALUES])
CRITERIA_PAIRED_VALUES = np.sort(CRITERIA_PAIRED_VALUES)
CRITERIA_NAMES = np.array(['< -1', '== -1', '(-1, 0)', '== 0', '(0, 1)', '== 1', '> 1'])
CRITERIA_PAIRED_NAMES = np.array(["[" + CRITERIA_NAMES[i] + ", " + CRITERIA_NAMES[j]  + "]"
                                   for i in CRITERIA_VALUES for j in CRITERIA_VALUES])
NUM_CRITERIA = len(CRITERIA)

EncodingResult = collections.namedtuple('EncodingResult', ['index_dct', 'sorted_mat', 'encodings'])
    #  index_dct: dict: key is encoding_str, value is a list of indexes
    #  sorted_mat: np.ndarray: The columns as sorted array
    #  encodings: list-encoding: indexed by position in the collection, value is the encoding for the indexed array


class Encoding(object):

    def __init__(self, collection: np.ndarray)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
        """
        self.collection = collection
        self.num_row, self.num_column = np.shape(collection)
        #
        # Outputs
        #  encoding_nm: matrix of encodings (column) for each array in the collection (row).
        #  encodings: list of encodings index by position of array in the collection.
        self.encoding_nm, self.encodings = self._makeEncoding()
        # Sorted encodings of criteria for each array
        self.sorted_encoding_arr = np.sort(np.array(self.encodings))
        # Sorted list of unique encodings
        self.unique_encodings = list(set(self.encodings))
        self.unique_encodings.sort()
        # Construct the dictionary of array indices with the same enoding.
        self.encoding_dct:dict = {k: [] for k in self.unique_encodings}
        {self.encoding_dct[e].append(i) for i, e in enumerate(self.encodings)}
        #
        self.num_partition = len(self.encoding_dct)
        # Encodings for pairs of arrays
        self.adjacent_pair_encoding_nm = self._makeAdjacentPairEncodingNamedMatrix()
        self.all_pair_encoding_nm = self._makeAllPairEncodingNamedMatrix()

    def __repr__(self)->str:
        return self.encoding_nm.__repr__()
    
    def _encodeCollection(self)->np.ndarray:
        """
        Creates a matrix with values that indicate the criteria
        satisfied by the original collection. We refer to this matrix as the fundamental encoding matrix.
        Shape is num_row, num_criteria.

        Returns:
            np.ndarray
        """
        # Index is offset by 1
        matrices = [(i+1)*f(self.collection) for i, f in enumerate(CRITERIA)]
        result = matrices[0]
        for matrix in matrices[1:]:
            result += matrix   # This works because criteria are mutually exclusive
        return result - 1  # Eliminate offset by 1
    
    def _DeprecatedmakeEncodingMat(self)->Tuple[np.ndarray, List[int]]:
        """
        Constructs the encoding matrix and the encodings.
        encoding_mat: matrix of the count of criteria satisfied by values for the array;
            columns are the encoding criteria and rows are the arrays in the collection.
        encodings: list of encodings for each array in the collection.

        Returns:
            np.ndarray: The encoding matrix.
        """
        adj_fundamental_mat = self._encodeCollection()
        encoding_rows:list = []
        matrix_rows:list = []
        # Calculate the criteria counts for each array
        for icol in range(len(CRITERIA)):
            row = np.sum(adj_fundamental_mat == icol, axis=1)
            encoding_rows.append((ENCODING_VAL**icol)*row)
            matrix_rows.append(row)
        # Construct the encoding matrix
        encodings = np.sum(encoding_rows, axis=0)
        encoding_mat = np.transpose(np.array(matrix_rows))
        return encoding_mat, encodings
    
    def _makeEncoding(self)->Tuple[NamedMatrix, list]:
        """
        Constructs a matrix that is a table of counts of criteria for each row.

        Returns:
            NamedMatrix
        """
        fundamental_mat = self._encodeCollection()
        encoding_value_arr = np.repeat(CRITERIA_VALUES, self.num_row)
        encoding_value_mat = np.reshape(encoding_value_arr, (NUM_CRITERIA, self.num_row)).T
        adj_fundamental_mat = np.concatenate([encoding_value_mat, fundamental_mat], axis=1)
        # Calculate the criteria counts for each array
        rows = []
        for row in adj_fundamental_mat:
            _, counts = np.unique(row, return_counts=True)
            rows.append(counts)
        table_mat = np.array(rows)
        table_mat -= 1  # Account for adding CRITERIA_VALUES
        named_matrix = NamedMatrix(table_mat, range(self.num_row), CRITERIA_VALUES,
                                   column_labels=CRITERIA_NAMES)
        # Encodings for each row
        encoding_powers = np.array([ENCODING_VAL**i for i in range(NUM_CRITERIA)])
        encodings = list(table_mat.dot(encoding_powers))
        return named_matrix, encodings
    
    def _makePairEncodingNamedMatrix(self, pairs: List[Tuple[int, int]])->NamedMatrix:
        """Creates a matrix that describes the encoding of pairs of arrays in self.collection.
           Rows correspond to the pairs provided. Columns correspond to the encoding of the pairs
           in CRITERIA_PAIRED_VALUES.

        Args:
            pairs

        Returns:
            NamedMatrix: self.num_row, len(CRITERIA_PAIRED_VALUES)
        """
        fundamental_mat = self._encodeCollection()
        augmenteds:list = []
        for pair in pairs:
            arr1 = fundamental_mat[pair[0]]
            arr2 = fundamental_mat[pair[1]]
            arr = arr1*ENCODING_VAL + arr2
            new_row = list(arr)
            new_row.extend(CRITERIA_PAIRED_VALUES)
            _, counts = np.unique(new_row, return_counts=True)
            augmenteds.append(counts)
        mat = np.array(augmenteds)
        mat -= 1  # Account for adding CRITERIA_PAIRED_VALUES
        named_matrix = NamedMatrix(array=mat, row_ids=pairs, column_ids=CRITERIA_PAIRED_VALUES)
        return named_matrix
    
    def _makeAdjacentPairEncodingNamedMatrix(self)->NamedMatrix:
        """
        Returns:
            NamedMatrix
                row: pairs of indicies in the collection
                columns: counts for CRITERIA_PAIRED_VALUES
        """
        pairs = [(i, i+1) for i in range(self.num_row-1)]
        return self._makePairEncodingNamedMatrix(pairs)
    
    def _makeAllPairEncodingNamedMatrix(self)->NamedMatrix:
        """
        Constructs the encoding dictionary for all pairs of arrays. Results are sorted.

        Returns:
            NamedMatrix
        """
        iter = itertools.permutations(range(self.num_row), 2)
        pairs = list(iter)
        return self._makePairEncodingNamedMatrix(pairs)   # type: ignore
    
    def __eq__(self, other)->bool:
        """
        Checks if the two ArrayCollectionEncoding have the same encoding.

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if len(self.sorted_encoding_arr) == len(other.sorted_encoding_arr):
            return np.allclose(self.sorted_encoding_arr, other.sorted_encoding_arr)
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