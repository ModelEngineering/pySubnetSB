'''Representation of a two dimensional array.'''

import numpy as np
import itertools
from typing import Optional

class Matrix(object):

    def __init__(self, array: np.ndarray):
        """
        Args:
            arr (np.array): Stoichiometry matrix. Rows are reactions; columns are species.
        Instance variables:
            num_mat (int): Number of matrices.
            num_row (int): Number of rows.
            num_column (int): Number of columns.
        """
        self.values = array
        self.shape = np.shape(array)
        if len(self.shape) == 2:
            self.num_mat = 1
            self.num_row, self.num_column = np.shape(array)
        elif len(self.shape) == 1:
            self.num_mat = 1
            self.num_row, self.num_column = len(array), 1
        elif len(self.shape) == 3:
            self.num_mat, self.num_row, self.num_column = np.shape(array)
        else:
            self.num_mat, self.num_row, self.num_column = 0, 0, 0
        self._hash:Optional[int] = None  # Deferred execution to improve efficiency

    def __len__(self)->int:
        return self.num_row

    @property
    def hash(self)->int:
        # This is an order dependent hash
        if self._hash is None:
            self._hash = hash(str(self.values))
        return self._hash

    def __repr__(self)->str:
        return str(self.values)

    def __eq__(self, other)->bool:
        if not np.all(np.shape(self.values) == np.shape(other.values)):
            return False
        return np.all(self.values == other.values)  # type: ignore
    
    def isCompatible(self, other)->bool:
        """
        Check if the matrix is compatible with another matrix.
        Args:
            other (Matrix): Another matrix.
        Returns:
            bool: True if the matrix is compatible; False otherwise.
        """
        return bool(np.all(np.shape(self.values) == np.shape(other.values)))
    
    def copy(self)->'Matrix':
        """
        Create a copy of the Matrix.

        Returns:
            Matrix: A copy of the Matrix.
        """
        return Matrix(self.values.copy())
    
    def isPermutablyIdentical(self, other):
        """
        Check if the matrix is permutable with another matrix.
        Args:
            matrix (np.array): Stoichiometry matrix. Rows are reactions; columns are species.
        Returns:
            bool: True if the matrix is permutable; False otherwise.
        """
        row_perm = itertools.permutations(range(self.num_row))
        col_perm = itertools.permutations(range(self.num_column))
        for r in row_perm:
            for c in col_perm:
                if np.all(self.values[r][:, c] == other.matrix):
                    return True
        return False
    
    @classmethod
    def makeTrinaryMatrix(cls, num_row: int=3, num_column: int=2, prob0=1.0/3)->np.ndarray:
        """
        Make a trinary matrix with 0, 1, and -1. No row or column can have all zeros.
        Args:
            nrow (int): Number of rows.
            ncol (int): Number of columns.
        Returns:
            np.array: A trinary matrix.
        """
        prob_other = (1.0 - prob0)/2
        arr = [0, 1, -1]
        prob_arr = [prob0, prob_other, prob_other]
        for _ in range(100):
            matrix = np.random.choice(arr, size=(num_row, num_column), p=prob_arr)
            matrix_sq = matrix*matrix
            is_nozerorow = np.all(matrix_sq.sum(axis=1) > 0)
            is_nozerocol = np.all(matrix_sq.sum(axis=0) > 0)
            if is_nozerorow and is_nozerocol:
                return matrix
        raise RuntimeError('Cannot generate a trinary matrix.')

    def randomize(self):  # type: ignore
        """
        Randomly permutes the rows and columns of the matrix.

        Returns:
            FixedMatrix
        """
        array = self.values.copy()
        row_perm = np.random.permutation(self.num_row)
        col_perm = np.random.permutation(self.num_column)
        return Matrix(array[row_perm][:, col_perm])  # type: ignore