'''Structures the rows and columns of a matrix based on order independent properties of the rows and columns.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.
"""

from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore

import numpy as np


SEPARATOR = 1000 # Separates the counts in a single numbera
# Classification of a dimension of the matrix
#    vals: classification of arrays in the dimension by ascending value
#    idxs: indices of the array in the dimension of the matrix


class _DimensionClassification(object):

    def __init__(self, arr: np.ndarray, idxs: np.ndarray):
        self.arr = arr
        self.idxs = idxs

    def __repr__(self)->str:
        return str(self.arr[self.idxs])


class OrderedMatrix(Matrix):
        
    def __init__(self, arr: np.ndarray):
        super().__init__(arr)
        if (self.nrow > SEPARATOR) or (self.ncol > SEPARATOR):
            raise ValueError("Matrix is too large to classify. Maximum number of rows, columns is 1000.")
        # Outputs
        self.row_classification = self.classifyRows()
        self.col_classification = self.classifyColumns()
        hash_arr = np.concatenate((self.row_classification.arr, self.col_classification.arr))
        self.hash_val = hashArray(hash_arr)

    def __repr__(self)->str:
        return str(self.arr) + '\n' + str(self.row_classification) + '\n' + str(self.col_classification)
    
    @staticmethod
    def classifyArray(arr: np.ndarray)->int:
        return np.sum(arr < 0) + np.sum(arr == 0) * SEPARATOR + np.sum(arr > 0)*SEPARATOR**2
    
    def classifyRows(self):
        """"
        Classifies the entries in a row.
        """
        classes = []
        for idx in range(self.nrow):
            classes.append(self.classifyArray(self.arr[idx, :]))
        sorted_classes = np.sort(classes)
        sorted_class_idxs = np.argsort(classes)
        return _DimensionClassification(sorted_classes, sorted_class_idxs)
    
    def classifyColumns(self):
        """"
        Classifies the entries in a row.
        """
        classes = []
        for idx in range(self.ncol):
            classes.append(self.classifyArray(self.arr))
        sorted_classes = np.sort(classes)
        sorted_class_idxs = np.argsort(classes)
        return _DimensionClassification(sorted_classes, sorted_class_idxs)
    
    def isCompatible(self, other)->bool:
        """
        Checks if the two matrices have the same DimensionClassifications for their rows and columns

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        is_true = np.allclose(self.row_classification.arr, other.row_classification.arr)
        is_true = is_true and np.allclose(self.col_classification.arr, other.col_classification.arr)
        return is_true