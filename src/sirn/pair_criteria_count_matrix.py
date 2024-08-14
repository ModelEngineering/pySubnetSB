'''A PairCriteriaCountMatrix counts of co-occurrences of criteria pairs satisfied by pairs of rows.'''

"""
To do:
1. Change interface to checking compatibility so returns matrices to compare rather than indices.
2. Need to produce matrices with # criteria**2 columns
"""

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.criteria_count_matrix import CriteriaCountMatrix # type: ignore
from sirn.named_matrix import Matrix  # type: ignore

import collections
import itertools
import numpy as np
from typing import Tuple, Optional


PairCriteriaResult = collections.namedtuple('PairCriteriaResult', 'row_offsets criteria_count_arr')

class PairCriteriaCountMatrix(CriteriaCountMatrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
            criteria_count_matrix (CriteriaCountMatrix): A matrix of criteria counts.
        """
        if criteria_vector is None:
            criteria_vector = CriteriaVector()
        pair_matrix_result = self._makePairCriteriaCountMatrix(array, criteria_vector)
        super().__init__(pair_matrix_result.criteria_count_arr, criteria_vector=criteria_vector)
        self._reference_matrix: Optional[Matrix] = None

    def _makePairCriteriaCountMatrix(self, values:np.ndarray, criteria_vec:CriteriaVector)->PairCriteriaResult:
        """
        Constructs 3-d matrix that describes pairwise comparisons of rows and criteria.
        Args:
            array (np.array): An array of real numbers.
        Returns:
            np.array: A matrix of boolean values.
        """
        num_row, num_column = values.shape
        row_perm = np.array(range(num_row))
        criteria_perm = np.array(range(criteria_vec.num_criteria))
        base_arrs = np.array([c(values) for c in criteria_vec.criteria_functions])
        row_offsets:list = []
        pair_matrices:list = []
        for row_idx in range(num_row):
            criteria_arrs:list = []
            for criteria_idx in range(criteria_vec.num_criteria):
                perm_arrs = base_arrs[criteria_perm, :]
                perm_arrs = perm_arrs[:, row_perm, :]
                combined_arr = np.logical_and(base_arrs, perm_arrs)
                # Aggregate across the rows
                sum_arr = np.sum(combined_arr, axis=2).T
                criteria_arrs.append(sum_arr)
                row_offsets.append(row_idx)
                criteria_perm = np.roll(criteria_perm, 1)
            pair_matrices.append(np.concatenate(criteria_arrs, axis=1))
            row_perm = np.roll(row_perm, 1)
        criteria_count_arr = np.array(pair_matrices)
        return PairCriteriaResult(row_offsets=row_offsets, criteria_count_arr=criteria_count_arr)
    
    ############## OVERRIDE PARENT METHODS #######3
    def getReferenceArray(self, assignment_len:Optional[int]=None)->np.ndarray:
        """
        Get a reference matrix of criteria counts against which another matrix is compared.

        Args:
            assignment_len (int): Number of rows in the matrix to be compared

        Returns:
            np.array: Matrix of counts to compare
        """
        if assignment_len is None:
            assignment_len = self.num_row
        if self._reference_matrix is None:
            assignment = np.array(range(self.num_row))
            assignment = assignment[:assignment_len]
            self._reference_matrix = self.getTargetArray(assignment)
        return self._reference_matrix

    def getTargetArray(self, assignment:np.ndarray[int])->np.ndarray:
        """
        Get a matrix as specified by the assignment. self.values is a 3-d array:
            1. Matrix selected
            2. Row in matrix
            3. Column in matrix

        Args:
            other_assignment (np.ndarray): Sequence of rows in original matrix that are to be compared

        Returns:
            np.ndarray: _description_
        """
        pairs = [(assignment[i], assignment[i+1]) for i in range(len(assignment)-1)]
        # Construct the indices to reference the desired rows
        array_indices = [ (pair[1] - pair[0] + self.num_row) % self.num_row for pair in pairs]
        return self.values[array_indices, np.array(assignment[:-1]), :]