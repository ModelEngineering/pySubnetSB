'''A CriteriaCountMatrix counts of occurrences of criteria satisfied in each row.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.criteria_count_matrix import CriteriaCountMatrix # type: ignore
from sirn.named_matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore

import numpy as np
from typing import List, Union, Optional

class SingleCriteriaCountMatrix(CriteriaCountMatrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
        """
        if criteria_vector is None:
            criteria_vector = CriteriaVector()
        values = self._makeCriteriaCountMatrix(array, criteria_vector)
        super().__init__(values, criteria_vector=criteria_vector)

    def __repr__(self)->str:
        named_matrix = NamedMatrix(self.values, row_name="rows", column_name="criteria")
        return named_matrix.__repr__()

    def _makeCriteriaCountMatrix(self, array:np.ndarray, criteria_vec:CriteriaVector)->np.ndarray:
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
    
    def isEqualValues(self, other)->bool:
        """
        Compare the values of two matrices.
        Args:
            other (CriteriaCountMatrix): Another matrix with same shape.
            max_permutation (int): The maximum number of permutations.
        Returns:
            bool: True if the values are equal.
        """
        return bool(np.all(self.values == other.values))
    
    def isLessEqualValues(self, other)->bool:
        """
        Compare the values of two matrices.
        Args:
            other (CriteriaCountMatrix): Another matrix with same shape.
            max_permutation (int): The maximum number of permutations.
        Returns:
            bool: True if the values are equal.
        """
        return bool(np.all(self.values <= other.values))