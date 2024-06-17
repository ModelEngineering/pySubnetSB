'''A PMatrix is a Matrix with capabilities for permuting rows and columns.'''
"""
To make operations more computationally efficient, rows and columns are ordered by
order independent properties.
"""

from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn.array_collection import ArrayCollection # type: ignore

import numpy as np
from typing import List, Optional


####################################
class PMatrix(Matrix):
        
    def __init__(self, array: np.ndarray,
                 row_names:Optional[List[str]]=None,
                 column_names:Optional[List[str]]=None,
                 model_name:Optional[str]=None): 
        # Inputs
        super().__init__(array)
        if row_names is None:
            row_names = [str(i) for i in range(self.num_row)]  # type: ignore
        if column_names is None:
            column_names = [str(i) for i in range(self.num_column)]  # type: ignore
        if model_name is None:
            model_name = str(np.random.randint(1000000))
        self.row_names = row_names
        self.column_names = column_names
        self.model_name = model_name
        # Outputs
        self.row_collection = ArrayCollection(self.array)
        column_arr = np.transpose(self.array)
        self.column_collection = ArrayCollection(column_arr)
        hash_arr = np.concatenate((self.row_collection.encoding_arr, self.column_collection.encoding_arr))
        self.hash_val = hashArray(hash_arr)

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
        if not self.model_name == other.model_name:
            return False
        return True

    def __repr__(self)->str:
        return str(self.array) + '\n' + str(self.row_collection) + '\n' + str(self.column_collection)
    
    def isPermutablyIdentical(self, other) -> bool:  # type: ignore
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
            return False
        # The matrices have the same shape and partitions
        #  Order the other PMatrix to align the partitions of the two matrices
        other_row_itr = other.row_collection.partitionPermutationIterator()
        other_column_itr = other.column_collection.partitionPermutationIterator()
        other_row_perm = next(other_row_itr)
        other_column_perm = next(other_column_itr)
        other_pmatrix = other.array[other_row_perm, :]
        other_pmatrix = other_pmatrix[:, other_column_perm]
        # Search all partition constrained permutations of this matrix to match the other matrix
        row_itr = self.row_collection.partitionPermutationIterator()
        count = 0
        array = self.array.copy()
        for row_perm in row_itr:
            column_itr = self.column_collection.partitionPermutationIterator()
            for col_perm in column_itr:
                count += 1
                matrix = array[row_perm, :]
                matrix = matrix[:, col_perm]
                if np.all(matrix == other_pmatrix):
                    return True
        return False
    
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