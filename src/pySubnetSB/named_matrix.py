'''Numpy 2 dimensional array with information about rows and columns.'''
from pySubnetSB.matrix import Matrix  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import collections
import json
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Union, List, Any


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
            if len(self._row_names) != np.shape(self.values)[0]:
                raise ValueError("Row names must have the same number of elements as the number of rows")
            self._row_names = np.array(self._row_names)
        return self._row_names  # type: ignore
    
    @property
    def column_names(self)->np.ndarray:
        if self._column_names is None:
            self._column_names = np.array([str(n) for n in range(self.num_column)])
        else:
            if len(self._column_names) != np.shape(self.values)[1]:
                raise ValueError("Column names must have the same number of elements as the number of columns")
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
        def findIndices(matrix: np.ndarray)->np.ndarray[int]:  # type: ignore
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
        if row_names is None:
            row_idxs = np.array(range(self.num_row))
        else:
            row_idxs = findIndices(row_names, self.row_names)
        if column_names is None:
            column_idxs = np.array(range(self.num_column))
        else:
            column_idxs = findIndices(column_names, self.column_names)
        new_values = self.values[row_idxs, :].copy()
        new_values = new_values[:, column_idxs]
        named_matrix = NamedMatrix(new_values, row_names=self.row_names[row_idxs],
                                   column_names=self.column_names[column_idxs])
        return SubsetResult(named_matrix=named_matrix,
                            row_idxs=row_idxs, column_idxs=column_idxs)
    
    def getRowIdx(self, row_name:str)->int:
        """
        Obtain the index of the row by name.

        Args:
            row_name (str): _description_

        Returns:
            int: _description_
        """
        return list(self.row_names).index(row_name)
    
    def getColumnIdx(self, column_name:str)->int:
        """
        Obtain the index of the column by name.

        Args:
            column_name (str): _description_

        Returns:
            int: _description_
        """
        return list(self.column_names).index(column_name)
    
    def getValueByNames(self, row_name:str, column_name:str)->Any:
        """
        Obtains the value of the NamedMatrix by row and column names.

        Args:
            row_name (str): _description_
            column_name (str): _description_

        Returns:
            Any: _description_
        """
        return self.values[self.getRowIdx(row_name), self.getColumnIdx(column_name)]
        
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
    
    def serialize(self)->str:
        """Serializes the NamedMatrix."""
        return json.dumps({cn.S_ID: str(self.__class__),
                           cn.S_ROW_NAMES: self.row_names.tolist(),
                           cn.S_COLUMN_NAMES: self.column_names.tolist(),
                           cn.S_ROW_DESCRIPTION: self.row_description,
                           cn.S_COLUMN_DESCRIPTION: self.column_description,
                           cn.S_VALUES: self.values.tolist()})
    
    @classmethod
    def deserialize(cls, string:str)->'NamedMatrix':
        """
        Deserializes the NamedMatrix.

        Args:
            string (str): A JSON string.

        Returns:
            NamedMatrix: A NamedMatrix object.
        """
        dct = json.loads(string)
        if not str(cls) in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        return cls(np.array(dct[cn.S_VALUES]), row_names=np.array(dct[cn.S_ROW_NAMES]),
                   column_names=np.array(dct[cn.S_COLUMN_NAMES]),
                   row_description=dct[cn.S_ROW_DESCRIPTION],
                   column_description=dct[cn.S_COLUMN_DESCRIPTION])
    
    @classmethod
    def hstack(cls, named_matrices:List['NamedMatrix'], is_rename:bool=False)->'NamedMatrix':
        """
        Stacks horizontally a list of of compatible NamedMatrix objects.

        Args:
            named_matrices (List[NamedMatrix])
            is_rename: bool (rename the columns)

        Returns:
            NamedMatrix

        Raises:
            ValueError: If the row names are not the same. 
        """
        # Check compatibility
        first_num_row = named_matrices[0].num_row
        first_row_names = named_matrices[0].row_names
        first_row_description = named_matrices[0].row_description
        first_column_description = named_matrices[0].column_description
        for named_matrix in named_matrices[1:]:
            if first_column_description != named_matrix.column_description:
                if len(first_column_description) > 0 and len(named_matrix.column_description) > 0:
                    raise ValueError("Column descriptions must be the same!")
            if first_row_description != named_matrix.row_description:
                if len(first_row_description) > 0 and len(named_matrix.row_description) > 0:
                    raise ValueError("Row descriptions must be the same!")
            if not np.all(first_row_names == named_matrix.row_names):
                raise ValueError("Row descriptions must be the same!")
            if first_num_row != named_matrix.num_row:
                raise ValueError("Number of rows must be the same!")
        # Stack the values and construct the NamedMatrix
        values = np.hstack([named_matrix.values for named_matrix in named_matrices])
        if is_rename:
            column_names = None
        column_names = np.concatenate([named_matrix.column_names for named_matrix in named_matrices])
        return NamedMatrix(values, row_names=first_row_names, column_names=column_names,
                           row_description=first_row_description,
                           column_description=first_column_description)
    
    @classmethod
    def vstack(cls, named_matrices:List['NamedMatrix'], is_rename:bool=False)->'NamedMatrix':
        """
        Stacks vertically a list of of compatible NamedMatrix objects.

        Args:
            named_matrices (List[NamedMatrix])

        Returns:
            NamedMatrix

        Raises:
            ValueError: If the row names are not the same. 
        """
        # Check compatibility
        first_num_column = named_matrices[0].num_column
        first_column_names = named_matrices[0].column_names
        first_column_description = named_matrices[0].column_description
        first_row_description = named_matrices[0].column_description
        for named_matrix in named_matrices[1:]:
            if first_row_description != named_matrix.row_description:
                raise ValueError("Row descriptions must be the same!")
            if first_column_description != named_matrix.column_description:
                raise ValueError("Column descriptions must be the same!")
            if not np.all(first_column_names == named_matrix.column_names):
                raise ValueError("Column descriptions must be the same!")
            if first_num_column != named_matrix.num_column:
                raise ValueError("Number of columns must be the same!")
        # Stack the values and construct the NamedMatrix
        values = np.vstack([named_matrix.values for named_matrix in named_matrices])
        if is_rename:
            row_names = None
        else:
            row_names = np.concatenate([named_matrix.row_names for named_matrix in named_matrices])
        return NamedMatrix(values, column_names=first_column_names, row_names=row_names,
                           column_description=first_column_description,
                           row_description=first_row_description)
    
    @classmethod
    def makeRandom(cls, num_row:int, num_column:int)->'NamedMatrix':
        """
        Create a random NamedMatrix.

        Args:
            num_row (int): Number of rows
            num_column (int): Number of columns

        Returns:
            NamedMatrix: A random NamedMatrix.
        """
        return NamedMatrix(np.random.rand(num_row, num_column))


NULL_NMAT = NamedMatrix(np.array([[]]))