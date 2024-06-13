'''Generates, Analyzes, and Manipulates Stoichiometry Matrices'''

import numpy as np
import itertools
from typing import Tuple

class Matrix(object):

    def __init__(self, array: np.ndarray):
        """
        Args:
            arr (np.array): Stoichiometry matrix. Rows are reactions; columns are species.
        """
        self.array = array
        self.num_row, self.num_column = np.shape(array)

    def __repr__(self)->str:
        return str(self.array)

    def __eq__(self, other)->bool:
        if np.shape(self.array) != np.shape(other.matrix):
            return False
        return np.all(self.array == other.matrix)  # type: ignore
    
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
                if np.all(self.array[r][:, c] == other.matrix):
                    return True
        return False
    
    @classmethod
    def makeTrinaryMatrix(cls, nrow: int=3, ncol: int=2, prob0=1.0/3)->np.array:
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
            matrix = np.random.choice(arr, size=(nrow, ncol), p=prob_arr)
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
        row_perm = np.random.permutation(self.num_row)
        col_perm = np.random.permutation(self.num_column)
        return Matrix(self.array[row_perm][:, col_perm])  # type: ignore