'''Common code for CriteriaCountMatrix counts of occurrences of criteria satisfied in each row.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.named_matrix import Matrix  # type: ignore
import sirn.util as util # type: ignore

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


HASH_BASE = np.int64(100) # Base of exponent used to separate encoding values, the count of criteria occurrences


class CriteriaCountMatrix(Matrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
        """
        self.criteria_vec = criteria_vector
        super().__init__(array)

#    def _sortMatrix(self)->Matrix:
#        """
#        Sort the rows of the matrix by their hash values.
#
#        Returns:
#            Matrix: _description_
#        """
#        def sort2dArray(arrays):
#            row_hashes = self.getRowHashes(arrays)
#            #sort_idx = np.argsort(row_hashes)
#            #sorted_arrs = arrays[sort_idx, :]
#            #return np.array(sorted_arrs)
#            result = np.sort(row_hashes)
#            return result
#        #
#        if self.num_mat == 1:
#            sorted_arr = sort2dArray(self.values)
#        else:
#            # Iterate across the 2d arrays
#            sorted_arrays = []
#            for arrays in self.values:
#                sorted_arr = sort2dArray(arrays)
#                sorted_arrays.append(sorted_arr)
#            sorted_arr = np.array(sorted_arrays)
#        return Matrix(sorted_arr)

    @staticmethod
    def getRowHashes(array)->np.ndarray:
        """
        Constructs a hash for each row of the matrix. The array returned is ordered by the rows.

        Returns:
            np.ndarray[int]: A list of hashes.
        """
        row_hashes = []
        for array in array:
            #row_hash = np.sum([np.int64(v)*HASH_BASE**n for n, v in enumerate(array)])
            row_hash = util.makeRowOrderIndependentHash(array)
            row_hashes.append(row_hash)
        return np.array(row_hashes)

    def __eq__(self, other)->bool:
        if not self.isCompatible(other):
            return False
        if not bool(np.all(self.criteria_vec == other.criteria_vec)):
            return False
        return bool(np.all(self.values == other.values))
    
    def copy(self)->'CriteriaCountMatrix':
        """
        Create a copy of the CriteriaCountMatrix.

        Returns:
            CriteriaCountMatrix: A copy of the CriteriaCountMatrix.
        """
        new_cc_mat = CriteriaCountMatrix(self.values.copy(), criteria_vector=self.criteria_vec)
        return new_cc_mat

    @abstractmethod 
    def getReferenceArray(self)->np.ndarray:
        raise NotImplementedError("Subclasses must implement.")

    @abstractmethod 
    def getTargetArray(self, assignment:np.ndarray[int])->np.ndarray:
        raise NotImplementedError("Subclasses must implement.")
    
    def isEqual(self, other:'CriteriaCountMatrix', other_assignment:np.ndarray)->bool:
        """
        Compare the values of two matrices.
        Args:
            other (CriteriaCountMatrix): Another matrix with same shape.
            other_assignment (np.ndarray): The assignment of other rows to self rows.
        Returns:
            bool: True if the values are equal.
        """
        return bool(np.all(self.getReferenceArray() == other.getTargetArray(other_assignment)))
    
    def isLessEqual(self, other:'CriteriaCountMatrix', other_assignment:np.ndarray)->bool:
        """
        Compare the values of two matrices.
        Args:
            other (CriteriaCountMatrix): Another matrix with same shape.
            other_assignment (np.ndarray): The assignment of other rows to self rows.
        Returns:
            bool: True if the values are equal.
        """
        return bool(np.all(self.getReferenceArray() <= other.getTargetArray(other_assignment)))