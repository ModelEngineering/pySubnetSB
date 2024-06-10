'''Structures the rows and columns of a matrix based on order independent properties of the rows and columns.'''


from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn.array_collection import ArrayCollection # type: ignore

import numpy as np


class OrderedMatrix(Matrix):
        
    def __init__(self, arr: np.ndarray):
        super().__init__(arr)
        # Outputs
        self.row_collection = ArrayCollection(self.arr)
        self.column_collection = ArrayCollection(np.transpose(self.arr))
        hash_arr = np.concatenate((self.row_collection.encoding, self.column_collection.encoding))
        self.hash_val = hashArray(hash_arr)

    def __repr__(self)->str:
        return str(self.arr) + '\n' + str(self.row_collection) + '\n' + str(self.column_collection)
    
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