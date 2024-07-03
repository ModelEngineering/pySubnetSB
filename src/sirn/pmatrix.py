'''A PMatrix is a Matrix with capabilities for permuting rows and columns.'''
"""
To make operations more computationally efficient, rows and columns are ordered by
order independent properties.
"""

from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn.array_collection import ArrayCollection # type: ignore
from sirn import constants as cn  # type: ignore

import collections
import itertools
import numpy as np
from typing import List, Optional



####################################
RandomizeResult = collections.namedtuple('RandomizeResult', ['pmatrix', 'row_perm', 'column_perm'])    


class PermutablyIdenticalResult(object):
    # Auxiliary object returned by isPermutablyIdentical

    def __init__(self, is_permutably_identical:bool,
                 is_compatible:bool=False,
                 is_excessive_perm:bool=False,
                 this_row_perms:Optional[List[np.ndarray]]=None,
                 this_column_perms:Optional[List[np.ndarray]]=None,
                 other_row_perm:Optional[np.ndarray[int]]=None,
                 other_column_perm:Optional[np.ndarray[int]]=None,
                 num_perm:int=0,
                 ):
        """
        Args:
            is_permutably_identical (bool): _description_
            is_compatible (bool): the two PMatrix have the same row and column encodings and counts
                                  wihin the encodings
            is_excessive_perm(bool): the number of permutations exceeds a threshold
            this_row_perms (Optional[List[int]], optional): Permutation of this matrix rows
            this_column_perms (Optional[List[int]], optional): Permutation of this matrix columns
            other_row_perm (Optional[List[int]], optional): Permutation of other matrix rows
            other_column_perm (Optional[List[int]], optional): Permutation of other matrix columns
            num_perm (Optional[int], optional): Number of permutations explored
        """
        self.is_excessive_perm = is_excessive_perm
        self.is_permutably_identical = is_permutably_identical
        if self.is_permutably_identical:
            is_compatible = True
        self.is_compatible = is_compatible
        if this_row_perms is None:
            this_row_perms = []
        if this_column_perms is None:
            this_column_perms = []
        self.this_row_perms = this_row_perms
        self.this_column_perms = this_column_perms
        self.other_row_perm = other_row_perm
        self.other_column_perm = other_column_perm
        self.num_perm = num_perm

    # Boolean value is the result of the test
    def __bool__(self)->bool:
        return self.is_permutably_identical


class PMatrix(Matrix):
        
    def __init__(self, array: np.ndarray,
                 row_names:Optional[List[str]]=None,
                 column_names:Optional[List[str]]=None,
                 ):
        """
        Abstraction for a permutable matrix

        Args:
            array (np.ndarray): _description_
            row_names (Optional[List[str]], optional): _description_. Defaults to None.
            column_names (Optional[List[str]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        # Inputs
        super().__init__(array)
        if row_names is None:
            row_names = [str(i) for i in range(self.num_row)]  # type: ignore
        if column_names is None:
            column_names = [str(i) for i in range(self.num_column)]  # type: ignore
        if len(row_names) != self.num_row:
            raise ValueError(f"Row names {len(row_names)} != {self.num_row}")
        self.row_names = row_names
        if len(column_names) != self.num_column:
            raise ValueError(f"Column names {len(column_names)} != {self.num_column}")
        self.column_names = column_names
        # Outputs
        self.row_collection = ArrayCollection(self.array)
        column_arr = np.transpose(self.array)
        self.column_collection = ArrayCollection(column_arr)
        hash_arr = np.concatenate((self.row_collection.encoding_arr, self.column_collection.encoding_arr))
        self.hash_val = hashArray(hash_arr)
        # log10 of the estimated number of permutations of rows and columns
        self.log_estimate = self.row_collection.log_estimate + self.column_collection.log_estimate

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
    

    def isPermutablyIdentical(self, other:'PMatrix', max_num_perm:int=cn.MAX_NUM_PERM,
            is_sirn:bool=True,
            is_find_all_perms:bool=True) -> PermutablyIdenticalResult:
        """
        Check if the matrices are permutably identical. Choose the correct algorithm.
        Order other PMatrix

        Args:
            other (PMatrix)
            max_num_perm (int): Maximum number of permutations to search
            is_sirn (bool): If True, use the SIRN algorithm
            is_find_all_perms (bool): If True, find all permutations that make the matrices equal
        Returns:
            bool
        """
        if is_sirn:
            return self._isPermutablyIdenticalSIRN(other, max_num_perm, is_find_all_perms)
        return self._isPermutablyIdenticalNotSirn(other, max_num_perm, is_find_all_perms) 
    
    def _isPermutablyIdenticalSIRN(self, other:'PMatrix', max_num_perm:int=cn.MAX_NUM_PERM,
                              is_find_all_perms:bool=True) -> PermutablyIdenticalResult:
        """
        Check if the matrices are permutably identical using the SIRN algorithm.
        Order other PMatrix

        Args:
            other (PMatrix)
            max_num_perm (int): Maximum number of permutations to search
            is_find_all_perms (bool): If True, find all permutations that make the matrices equal
        Returns:
            bool
        """
        # Check compatibility
        if not self.isCompatible(other):
            return PermutablyIdenticalResult(False)
        # The matrices have the same shape and partitions
        #  Order the other PMatrix to align the partitions of the two matrices
        other_row_perm =  list(other.row_collection.partitionPermutationIterator())[0]
        other_column_perm = list(other.column_collection.partitionPermutationIterator())[0]
        other_array = self.permuteArray(other.array, other_row_perm, other_column_perm)
        # Search all partition constrained permutations of this matrix to match the other matrix
        row_itr = self.row_collection.partitionPermutationIterator()
        num_perm = 0
        # Find all permutations that result in equality
        this_row_perms: list = []
        this_column_perms: list = []
        is_done = False
        is_excessive_perm = False
        for row_perm in row_itr:
            if is_done:
                break
            # There may be a large number of column permutations
            column_itr = self.column_collection.partitionPermutationIterator()
            for column_perm in column_itr:
                if num_perm >= max_num_perm:
                    is_excessive_perm = True
                    is_done = True
                    break
                num_perm += 1
                this_array = self.permuteArray(self.array, row_perm, column_perm)
                if np.all(this_array == other_array):
                    this_row_perms.append(row_perm)
                    this_column_perms.append(column_perm)
                    if not is_find_all_perms:
                        break
        #
        is_permutably_identical = len(this_row_perms) > 0
        permutably_identical_result = PermutablyIdenticalResult(is_permutably_identical,
                            this_row_perms=this_row_perms, this_column_perms=this_column_perms,
                            other_row_perm=other_row_perm, other_column_perm=other_column_perm,
                            is_excessive_perm=is_excessive_perm, num_perm=num_perm)
        return permutably_identical_result
     
    def _isPermutablyIdenticalNotSirn(self, other:'PMatrix', max_num_perm:int=cn.MAX_NUM_PERM,
                              is_find_all_perms:bool=True) -> PermutablyIdenticalResult:
        """
        Check if the matrices are permutably by considering all permuations.

        Args:
            other (PMatrix)
            max_num_perm (int): Maximum number of permutations to search
            is_find_all_perms (bool): If True, find all permutations that make the matrices equal
        Returns:
            bool
        """
        # Check compatibility
        this_shape = np.shape(self.array)
        other_shape = np.shape(other.array)
        if this_shape != other_shape:
            return PermutablyIdenticalResult(False)
        # 
        num_row = this_shape[0]
        num_column = this_shape[1]
        # Search all permutations of this matrix to match the other matrix
        other_row_perm = np.array(range(num_row))
        other_column_perm = np.array(range(num_column))
        other_array = self.permuteArray(other.array, other_row_perm, other_column_perm)
        this_row_perms: list = []
        this_column_perms: list = []
        is_done = False
        num_perm = 0
        row_itr = itertools.permutations(range(num_row))
        for row_perm in row_itr:
            if is_done:
                break
            column_itr = itertools.permutations(range(num_column))
            for column_perm in column_itr:
                if is_done:
                    break
                this_array = self.permuteArray(self.array, np.array(row_perm), np.array(column_perm))
                num_perm += 1
                if np.all(this_array == other_array):
                    this_row_perms.append(row_perm)
                    this_column_perms.append(column_perm)
                    if not is_find_all_perms:
                        is_done = True
                        break
                if num_perm >= max_num_perm:
                    is_done = True
                    break
        is_permutably_identical = len(this_row_perms) > 0
        is_excessive_perm = num_perm >= max_num_perm
        permutably_identical_result = PermutablyIdenticalResult(is_permutably_identical,
                            this_row_perms=this_row_perms, this_column_perms=this_column_perms,
                            other_row_perm=other_row_perm, other_column_perm=other_column_perm,
                            is_excessive_perm=is_excessive_perm, num_perm=num_perm)
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