'''A CriteriaCountMatrix counts of occurrences of criteria satisfied in each row.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.criteria_count_matrix import CriteriaCountMatrix # type: ignore
from sirn.named_matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore
from sirn import util # type: ignore

import numpy as np
from typing import Optional

class SingleCriteriaCountMatrix(CriteriaCountMatrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
        """
        if criteria_vector is None:
            criteria_vector = CriteriaVector()
        values = self._makeSingleCriteriaCountMatrix(array, criteria_vector)
        super().__init__(values, criteria_vector=criteria_vector)

#    def _getRowHashes(self, array:np.ndarray)->np.ndarray:
#        """
#        Get a list of hashes for each row.
#        Returns:
#            List[int]: A list of hashes.
#        """
#        #return np.sort(self.getRowHashes(self.sorted_hash_arr.values))
#        return self.sorted_hash_arr

    def __repr__(self)->str:
        named_matrix = NamedMatrix(self.values, row_description="rows", column_description="criteria",
               column_names=self.criteria_vec.criteria_strs)  # type: ignore
        return named_matrix.__repr__()
    
    def getReferenceArray(self)->np.ndarray:
        """
        Get a reference matrix of criteria counts against which another matrix is compared.
        Returns:
            np.array: Matrix of counts to compare
        """
        return self.values

    def getTargetArray(self, assignment:np.ndarray[int])->np.ndarray:
        """
        Get a matrix as specified by the assignment.

        Args:
            other_assignment (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.values[assignment, :]

    def _makeSingleCriteriaCountMatrix(self, array:np.ndarray, criteria_vec:CriteriaVector)->np.ndarray:
        """
        Evaluate the criteria on an array.
        Args:
            array (np.array): An array of real numbers.
        Returns:
            np.array: A matrix of boolean values.
        """
        lst = [c(array.T) for c in criteria_vec.criteria_functions]  # type: ignore
        result = np.sum(lst, axis=1)
        return result.T