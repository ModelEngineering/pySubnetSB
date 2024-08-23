'''Numpy 2 dimensional array with information about rows and columns.'''
from sirn.matrix import Matrix  # type: ignore

import collections
from IPython.display import display
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Union


SubsetResult = collections.namedtuple('SubsetResult', ['named_matrix', 'row_idxs', 'column_idxs'])


class NamedMatrix(Matrix):

    def __init__(self, array: np.ndarray,
                 row_names:Optional[np.ndarray]=None,
                 column_names:Optional[np.ndarray]=None,
                 row_description:str = "",
                 column_description:str = ""):
        """

        Args:
            matrix (np.ndarray): 2d numpy array
            row_names (np.ndarray): convenient identifier for rows
            column_names (np.ndarray): convenient identifier for columns
            row_description (str, optional): Name of the row. Defaults to "". Name applied to the rows.
            column_description (str, optional): Name of the column. Defaults to "". Name applied to the columns.
        """
        # Most properties are assigned on first reference since a NamedMatrix may be used only
        # as a shallow container for np.ndarray
        super().__init__(array)
        self.row_description = row_description
        self.column_description = column_description
        self._row_names:Optional[np.ndarray] = row_names
        self._column_names:Optional[np.ndarray] = column_names
        # The following are deferred execution for efficiency considerations
        self._dataframe:Optional[pd.DataFrame] = None

    @property
    def row_names(self)->np.ndarray:
        if self._row_names is None:
            self._row_names = np.array([str(n) for n in range(self.num_row)])
        else:
            self._row_names = np.array(self._row_names)
        return self._row_names  # type: ignore
    
    @property
    def column_names(self)->np.ndarray:
        if self._column_names is None:
            self._column_names = np.array([str(n) for n in range(self.num_column)])
        else:
            self._column_names = np.array(self._column_names)
        return self._column_names  # type: ignore
    
    @property
    def dataframe(self)->pd.DataFrame:
        if self._dataframe is None:
            #reduced_named_matrix = self._deleteZeroRowsColumns()
            reduced_named_matrix = self
            if len(reduced_named_matrix.values) == 0:
                return pd.DataFrame()
            self._dataframe = pd.DataFrame(reduced_named_matrix.values)
            if len(reduced_named_matrix.row_names.shape) == 1:
                   self._dataframe.index = reduced_named_matrix.row_names
            if len(reduced_named_matrix.column_names.shape) == 1:
                   self._dataframe.columns = reduced_named_matrix.column_names
            self._dataframe.index.name = self.row_description
            self._dataframe.columns.name = self.column_description
        return self._dataframe
    
    def copy(self)->'NamedMatrix':
        """
        Create a copy of the NamedMatrix.

        Returns:
            NamedMatrix: A copy of the NamedMatrix.
        """
        return NamedMatrix(self.values.copy(), row_names=self.row_names.copy(),
                           column_names=self.column_names.copy(),
                           row_description=self.row_description, column_description=self.column_description)
    
    def __eq__(self, other):
        """
        Compare the properties of the two NamedMatrix objects.

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not super().__eq__(other):
            return False
        for attr in ['row_names', 'column_names', 'row_description', 'column_description']:
            if not np.all(getattr(self, attr) == getattr(other, attr)):
                return False
        return True

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
        row_idxs = findIndices(self.values)
        if len(row_idxs) == 0:
            return NamedMatrix(np.array([]), np.array([]), np.array([]))
        column_idxs = findIndices(self.values.T)
        matrix = self.values.copy()
        matrix = matrix[row_idxs, :]
        matrix = matrix[:, column_idxs]
        row_names = self.row_names[row_idxs]
        column_names = self.column_names[column_idxs]
        return NamedMatrix(matrix, row_names=row_names, column_names=column_names,
                           row_description=self.row_description, column_description=self.column_description)
    
    def template(self, matrix:Optional[Union[np.ndarray, Matrix]]=None)->'NamedMatrix':
        """
        Create a new NamedMatrix with the same row and column names but with a new matrix.

        Args:
            matrix (np.ndarray): New matrix to use. If None, then self is used.

        Returns:
            NamedMatrix: New NamedMatrix with the same row and column names but with a new matrix.
        """
        if matrix is None:
            matrix = self.values.copy()
        if isinstance(matrix, Matrix):
            matrix = matrix.values
        if not np.allclose(matrix.shape, self.values.shape):
            raise ValueError("Matrix shape must be the same as the original matrix")
        return NamedMatrix(matrix, row_names=self.row_names, column_names=self.column_names,
                            row_description=self.row_description, column_description=self.column_description)
    
    def isCompatible(self, other:'NamedMatrix')->bool:
        if not np.allclose(self.values.shape, other.values.shape):
            return False
        is_true =  np.all(self.row_names == other.row_names) and np.all(self.column_names == other.column_names) 
        return bool(is_true)
    
    def __repr__(self):
        return self.dataframe.__repr__()
    
    def __le__(self, other)->bool:
        if not self.isCompatible(other):
            return False
        return bool(np.all(self.values <= other.values))
        
    def getSubNamedMatrix(self, row_names:Optional[Union[np.ndarray, list]]=None,
                     column_names:Optional[Union[np.ndarray, list]]=None)->SubsetResult:
        """
        Create an ndarray that is a subset of the rows in the NamedMatrix.

        Args:
            row_names (list): List of row names to keep. If None, keep all.
            column_names (list): List of row names to keep. If None, keep all.

        Returns:
            SubsetResult (readonly values)
        """
        def cleanName(name):
            if name[0] in ["[", "("]:
                new_name = name[1:]
            else:
                new_name = name
            if name[-1] in ["]", ")"]:
                new_name = new_name[:-1]
            return new_name.replace(",", "")
        def findIndices(sub_names,
                        all_names=None)->np.ndarray:
            if all_names is None:
                return np.array(range(len(sub_names)))
            sub_names_lst = [cleanName(str(n)) for n in sub_names]
            all_names_lst = [cleanName(str(n)) for n in all_names]
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
        row_idxs = findIndices(row_names, self.row_names)  # type: ignore
        column_idxs = findIndices(column_names, self.column_names)  # type: ignore
        new_values = self.values[row_idxs, :].copy()
        new_values = new_values[:, column_idxs]
        named_matrix = NamedMatrix(new_values, row_names=self.row_names[row_idxs],
                                   column_names=self.column_names[column_idxs])
        return SubsetResult(named_matrix=named_matrix,
                            row_idxs=row_idxs, column_idxs=column_idxs)
        
    def getSubMatrix(self, row_idxs:np.ndarray, column_idxs:np.ndarray)->Matrix:
        """
        Create an ndarray that is a subset of the rows in the NamedMatrix.

        Args:
            row_idxs (ndarray): row indices to keep
            column_idxs (ndarray): column indices to keep

        Returns:
            Matrix
        """
        return Matrix(self.values[row_idxs, :][:, column_idxs])

    def transpose(self):
        return NamedMatrix(self.values.T, row_names=self.column_names,
                           column_names=self.row_names,
                           row_description=self.column_description, column_description=self.row_description)