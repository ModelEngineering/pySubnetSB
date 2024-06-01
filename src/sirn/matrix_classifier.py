'''Classifies the rows and columns of a matrix. Provides comparisons between matrices.'''

"""
Arrays are classified based on the number of values that are less than 0, equal to 0, and greater than 0.
"""

from src.sirn.classify_array import ArrayClassifier # type: ignore

import collections
import numpy as np


SEPARATOR = 1000 # Separates the counts in a single numbera
# Classification of a dimension of the matrix
#    vals: classification of arrays in the dimension by ascending value
#    idxs: indices of the array in the dimension of the matrix


class _DimensionClassification(object):

    def __init__(self, vals: np.ndarray, idxs: np.ndarray):
        self.vals = vals
        self.idxs = idxs

    def __repr__(self)->str:
        return str(self.vals[self.idxs])


class MatrixClassifier(object):
        
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.nrow, self.ncol = np.shape(arr)
        if (10**self.nrow > SEPARATOR) or (10**self.ncol > SEPARATOR):
            raise ValueError("Matrix is too large to classify. Maximum number of rows, columns is 1000.")
        # Outputs
        self.row_classification = self.classifyRows()
        self.col_classification = self.classifyColumns()

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
            classifier = ArrayClassifier(self.arr[:, idx])
            classes.append(ArrayClassifier(classifier.encoding))
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
        is_true = np.allclose(self.row_classification.vals, other.row_classification.vals)
        is_true = is_true and np.allclose(self.col_classification.vals, other.col_classification.vals)
        return is_true