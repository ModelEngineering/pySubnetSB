'''Common code for CriteriaCountMatrix counts of occurrences of criteria satisfied in each row.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.named_matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Optional

class CriteriaCountMatrix(Matrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
        """
        self.criteria_vec = criteria_vector
        super().__init__(array)

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