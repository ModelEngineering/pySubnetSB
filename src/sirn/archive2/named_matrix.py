'''Numpy 2 dimensional array with information about rows and columns.'''

import collections
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Tuple


SubsetResult = collections.namedtuple('SubsetResult', ['matrix', 'row_idxs', 'column_idxs'])


class NamedMatrix(object):

    def __init__(self, matrix: np.ndarray, row_ids: np.ndarray, column_ids: np.ndarray,
                 row_labels: Optional[np.ndarray[str]] = None, column_labels: Optional[np.ndarray[str]] = None):
        """

        Args:
            matrix (np.ndarray): 2d numpy array
            row_ids (np.ndarray): convenient identifier for rows
            column_ids (np.ndarray): convenient identifier for columns
            row_labels (Optional[np.ndarray[str]], optional): Human readable labels for rows. Defaults to None.
            column_labels (Optional[np.ndarray[str]], optional): Human readable labels for columns. Defaults to None.
        """
        # These initializations are done to handle an empty matrix
        self.matrix = matrix
        self.num_row = 0
        self.num_column = 0
        self.shape = self.matrix.shape
        self.row_ids = np.array([])
        self.column_ids = np.array([])
        self._dataframe = pd.DataFrame()  # DataFrame representation of the matrix
        if row_labels is None:
            row_labels = np.array([str(n) for n in row_ids])  # type: ignore
        if column_labels is None:
            column_labels = np.array([str(n) for n in column_ids])  # type: ignore
        self.row_labels = row_labels
        self.column_labels = column_labels
        #
        if len(self.shape) == 2:
            self.num_row, self.num_column = self.matrix.shape
            self.row_ids = np.array(row_ids)
            if (row_ids is not None) and (len(row_ids) != self.num_row):
                raise ValueError("Number of row ids must be equal to the number of rows in the matrix")
            self.column_ids = np.array(column_ids)
            if (column_ids is not None) and (len(column_ids) != self.num_column):
                raise ValueError("Number of column ids must be equal to the number of columns in the matrix")

    def _deleteZeroRowsColumns(self)->'NamedMatrix':
        """
        Delete rows and columns that are all zeros.

        Returns:
            NamedMatrix: New NamedMatrix with zero rows and columns removed.
        """
        def findIndices(matrix: np.ndarray)->np.ndarray[int]:
            # Finds inidices of non-zero rows
            indices = []   # Indices to delete
            for idx, array in enumerate(matrix):
                if not np.allclose(array, 0):
                    indices.append(idx)
            return np.array(indices)
        #
        row_idxs = findIndices(self.matrix)
        if len(row_idxs) == 0:
            return NamedMatrix(np.array([]), np.array([]), np.array([]))
        column_idxs = findIndices(self.matrix.T)
        matrix = self.matrix.copy()
        matrix = matrix[row_idxs, :]
        matrix = matrix[:, column_idxs]
        row_ids = self.row_ids[row_idxs]
        column_ids = self.column_ids[column_idxs]
        row_labels = self.row_labels[row_idxs]
        column_labels = self.column_labels[column_idxs]
        return NamedMatrix(matrix, row_ids, column_ids,
                           row_labels=row_labels, column_labels=column_labels)
    
    def template(self, matrix:Optional[np.ndarray]=None)->'NamedMatrix':
        """
        Create a new NamedMatrix with the same row and column names but with a new matrix.

        Args:
            matrix (np.ndarray): New matrix to use. If None, then self.matrix is used.

        Returns:
            NamedMatrix: New NamedMatrix with the same row and column names but with a new matrix.
        """
        if matrix is None:
            matrix = self.matrix.copy()
        if not np.allclose(matrix.shape, self.matrix.shape):
            raise ValueError("Matrix shape must be the same as the original matrix")
        return NamedMatrix(matrix, self.row_ids, self.column_ids,
                           row_labels=self.row_labels, column_labels=self.column_labels)
    
    def isCompatible(self, other:'NamedMatrix')->bool:
        if not np.allclose(self.matrix.shape, other.matrix.shape):
            return False
        is_true =  np.all(self.row_ids == other.row_ids) and np.all(self.column_ids == other.column_ids)  \
            and np.all(self.row_labels == other.row_labels) and np.all(self.column_labels == other.column_labels)
        return bool(is_true)
    
    @property
    def dataframe(self)->pd.DataFrame:
        if len(self._dataframe) == 0:
            reduced_named_matrix = self._deleteZeroRowsColumns()
            if len(reduced_named_matrix.matrix) == 0:
                return pd.DataFrame()
            self._dataframe = pd.DataFrame(reduced_named_matrix.matrix, index=reduced_named_matrix.row_labels,
                   columns=reduced_named_matrix.column_labels)
        return self._dataframe
    
    def __repr__(self):
        return str(self.dataframe.__repr__())
    
    def __eq__(self, other):
        """
        Compare the properties of the two NamedMatrix objects.

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not self.isCompatible(other):
            return False
        return bool(np.allclose(self.matrix, other.matrix))
    
    def __le__(self, other)->bool:
        if not self.isCompatible(other):
            return False
        return bool(np.all(self.matrix <= other.matrix))
        
    def getSubMatrix(self, row_ids:Optional[list]=None, column_ids:Optional[list]=None)->SubsetResult:
        """
        Create an ndarray that is a subset of the rows in the NamedMatrix.

        Args:
            row_ids (list): List of row names to keep. If None, keep all.
            column_ids (list): List of row names to keep. If None, keep all.

        Returns:
            SubsetResult (readonly values)
        """
        def findIndices(sub_names:np.ndarray, all_names:np.ndarray)->np.ndarray:
            if all_names is None:
                return np.range(len(sub_names))
            sub_names_lst = [str(n) for n in sub_names]
            all_names_lst = [str(n) for n in all_names]
            indices = np.repeat(-1, len(sub_names))
            # Find the indices of the names in the other_names and place them in the correct order
            for sub_idx, sub_name in enumerate(sub_names_lst):
                if any([np.all(sub_name == o) for o in all_names_lst]):
                    all_names_idx = all_names_lst.index(sub_name)
                    indices[sub_idx] = all_names_idx
                else:
                    raise ValueError(f'Could not find name {sub_name} in other names!')
            return np.array(indices)
        #
        if row_ids is None:
            row_idxs = np.array(range(self.num_row))
        else:
            row_idxs = findIndices(np.array(row_ids), self.row_ids)
        if column_ids is None:
            column_idxs = np.array(range(self.num_column))
        else:
            column_idxs = findIndices(np.array(column_ids), self.column_ids)  # type: ignore
        new_matrix = self.matrix[row_idxs, :].copy()
        new_matrix = new_matrix[:, column_idxs]
        return SubsetResult(matrix=new_matrix, row_idxs=row_idxs, column_idxs=column_idxs)
    
    def getSubNamedMatrix(self, row_ids:Optional[list]=None, column_ids:Optional[list]=None)->'NamedMatrix':
        """
        Create an ndarray that is a subset of the rows in the NamedMatrix. Return rows and columns
        sorted in the same order as specified by the arguments.

        Args:
            row_ids (list): List of row names to keep. If None, keep all.
            column_ids (list): List of row names to keep. If None, keep all.

        Returns:
            SubsetResult (readonly values)
        """
        def get(array, idxs):
            if idxs is None:
                return array
            return array[idxs]
        #
        subset_result = self.getSubMatrix(row_ids=row_ids, column_ids=column_ids)
        mat = subset_result.matrix.copy()
        row_ids = get(self.row_ids, subset_result.row_idxs)
        column_ids = get(self.column_ids, subset_result.column_idxs)
        row_labels = get(self.row_labels, subset_result.row_idxs)
        column_labels = get(self.column_labels, subset_result.column_idxs)
        return NamedMatrix(mat, row_ids, column_ids, row_labels=row_labels, column_labels=column_labels)  # type: ignore