'''A PMatrix is a Matrix with capabilities for permuting rows and columns.'''
"""
To make operations more computationally efficient, rows and columns are ordered by
order independent properties.
"""

from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn.array_collection import ArrayCollection # type: ignore

import collections
import numpy as np
from typing import List, Optional



####################################
RandomizeResult = collections.namedtuple('RandomizeResult', ['pmatrix', 'row_perm', 'column_perm'])    


class PermutablyIdenticalResult(object):
    # Auxiliary object returned by isPermutablyIdentical

    def __init__(self, is_permutably_identical:bool,
                 this_row_perms:Optional[List[np.ndarray]]=None,
                 this_column_perms:Optional[List[np.ndarray]]=None,
                 other_row_perm:Optional[np.ndarray[int]]=None,
                 other_column_perm:Optional[np.ndarray[int]]=None,
                 ):
        """
        Args:
            is_permutably_identical (bool): _description_
            this_row_perms (Optional[List[int]], optional): Permutation of this matrix rows
            this_column_perms (Optional[List[int]], optional): Permutation of this matrix columns
            other_row_perm (Optional[List[int]], optional): Permutation of other matrix rows
            other_column_perm (Optional[List[int]], optional): Permutation of other matrix columns
        """
        self.is_permutably_identical = is_permutably_identical
        if this_row_perms is None:
            this_row_perms = []
        if this_column_perms is None:
            this_column_perms = []
        self.this_row_perms = this_row_perms
        self.this_column_perms = this_column_perms
        self.other_row_perm = other_row_perm
        self.other_column_perm = other_column_perm

    # Boolean value is the result of the test
    def __bool__(self)->bool:
        return self.is_permutably_identical


class PMatrix(Matrix):
        
    def __init__(self, array: np.ndarray,
                 row_names:Optional[List[str]]=None,
                 column_names:Optional[List[str]]=None,
                 ): 
        # Inputs
        super().__init__(array)
        if row_names is None:
            row_names = [str(i) for i in range(self.num_row)]  # type: ignore
        if column_names is None:
            column_names = [str(i) for i in range(self.num_column)]  # type: ignore
        self.row_names = row_names
        self.column_names = column_names
        # Outputs
        self.row_collection = ArrayCollection(self.array)
        column_arr = np.transpose(self.array)
        self.column_collection = ArrayCollection(column_arr)
        hash_arr = np.concatenate((self.row_collection.encoding_arr, self.column_collection.encoding_arr))
        self.hash_val = hashArray(hash_arr)

    def randomize(self)->RandomizeResult:
        """Randomly permutes the rows and columns of the matrix.

        Returns:
            PMatrix
        """
        array = self.array.copy()
        for _ in range(10):
            row_perm = np.random.permutation(self.num_row)
            column_perm = np.random.permutation(self.num_column)
        array = self.permuteArray(array, row_perm, column_perm)
        randomize_result = RandomizeResult(
            pmatrix= PMatrix(array, self.row_names.copy(), self.column_names.copy()),
            row_perm=row_perm,
            column_perm=column_perm
            )
        return randomize_result
        

    def copy(self)->'PMatrix':
        return PMatrix(self.array.copy(), self.row_names.copy(), self.column_names.copy())

    def __eq__(self, other)->bool:
        """Check if two PMatrix have the same values

        Returns:
            bool: True if the matrix
        """
        if not super().__eq__(other):
            return False
        if not all([s == o] for s, o in zip(self.row_names, other.row_names)):
            return False
        if not all([s == o] for s, o in zip(self.column_names, other.column_names)):
            return False
        if not all([s == o] for s, o in zip(self.row_collection.encoding_arr,
                other.row_collection.encoding_arr)):
            return False
        if not all([s == o] for s, o in zip(self.column_collection.encoding_arr,
                other.column_collection.encoding_arr)):
            return False
        return True

    def __repr__(self)->str:
        return str(self.array) + '\n' + str(self.row_collection) + '\n' + str(self.column_collection)

    @staticmethod 
    def permuteArray(array:np.ndarray, row_perm:np.ndarray[int],
                     column_perm:np.ndarray[int])->np.ndarray:
        """
        Permute the rows and columns of a matrix.

        Args:
            array (np.array): Matrix to permute.
            row_perm (np.array): Permutation of the rows.
            column_perm (np.array): Permutation of the columns.

        Returns:
            np.array: Permuted matrix.
        """
        new_array = array.copy()
        new_array = new_array[row_perm, :]
        return new_array[:, column_perm]
    
    def isPermutablyIdentical(self, other:'PMatrix') -> PermutablyIdenticalResult:
        """
        Check if the matrices are permutably identical.
        Order other PMatrix

        Args:
            other (PMatrix)
        Returns:
            bool
        """
        # Check compatibility
        if not self.isCompatible(other):
            return PermutablyIdenticalResult(False)
        # The matrices have the same shape and partitions
        #  Order the other PMatrix to align the partitions of the two matrices
        other_row_itr = other.row_collection.partitionPermutationIterator()
        other_column_itr = other.column_collection.partitionPermutationIterator()
        other_row_perm = next(other_row_itr)
        other_column_perm = next(other_column_itr)
        other_array = self.permuteArray(other.array, other_row_perm, other_column_perm)
        # Search all partition constrained permutations of this matrix to match the other matrix
        row_itr = self.row_collection.partitionPermutationIterator()
        count = 0
        # Find all permutations that result in equality
        this_row_perms: list = []
        this_column_perms: list = []
        for row_perm in row_itr:
            column_itr = self.column_collection.partitionPermutationIterator()
            for column_perm in column_itr:
                count += 1
                this_array = self.permuteArray(self.array, row_perm, column_perm)
                if np.all(this_array == other_array):
                    this_row_perms.append(row_perm)
                    this_column_perms.append(column_perm)
        #
        is_permutably_identical = len(this_row_perms) > 0
        permutably_identical_result = PermutablyIdenticalResult(is_permutably_identical,
                            this_row_perms=this_row_perms, this_column_perms=this_column_perms,
                            other_row_perm=other_row_perm, other_column_perm=other_column_perm)
        return permutably_identical_result
    
    def isCompatible(self, other)->bool:
        """
        Checks if the two matrices have the same DimensionClassifications for their rows and columns

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        is_true = self.row_collection.isCompatible(other.row_collection)
        is_true = is_true and self.column_collection.isCompatible(other.column_collection)
        return is_true