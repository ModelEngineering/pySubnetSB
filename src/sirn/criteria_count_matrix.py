'''A CriteriaCountMatrix counts of occurrences of criteria satisfied in each row.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.matrix import Matrix  # type: ignore

import numpy as np
from typing import List, Union, Optional

class CriteriaCountMatrix(Matrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
        """
        if criteria_vector is None:
            criteria_vector = CriteriaVector()
        self.criteria_vec = criteria_vector.criteria_functions
        values = self._makeCriteriaCountMatrix(array)
        super().__init__(values)

    def _makeCriteriaCountMatrix(self, array:np.ndarray)->np.ndarray:
        """
        Evaluate the criteria on an array.
        Args:
            array (np.array): An array of real numbers.
        Returns:
            np.array: A matrix of boolean values.
        """
        lst = [c(array.T) for c in self.criteria_vec]  # type: ignore
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