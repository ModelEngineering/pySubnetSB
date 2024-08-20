'''A PairCriteriaCountMatrix counts of co-occurrences of criteria pairs satisfied by pairs of rows.'''


from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.criteria_count_matrix import CriteriaCountMatrix # type: ignore
from sirn.named_matrix import Matrix, NamedMatrix  # type: ignore

import collections
import itertools
import numpy as np
import pandas as pd # type: ignore
from typing import Tuple, Optional, List


class PairCriteriaCountMatrix(CriteriaCountMatrix):
    def __init__(self, array:np.array, criteria_vector:Optional[CriteriaVector]=None):
        """
        Args:
            array (np.array): An array of real numbers.
            criteria_vector (CriteriaVector): A vector of criteria.
            criteria_count_matrix (CriteriaCountMatrix): A matrix of criteria counts.
        """
        self.array = array
        if criteria_vector is None:
            criteria_vector = CriteriaVector()
        self.criteria_vector = criteria_vector
        criteria_count_arr, self.criteria_vector_idx_pairs = self._makePairCriteriaCountMatrix(array, criteria_vector)
        super().__init__(criteria_count_arr, criteria_vector=criteria_vector)
        self._reference_matrix: Optional[Matrix] = None

    def _makeColumnLabels(self, pairs=None)->List[str]:
        if pairs is None:
            pairs = self.criteria_vector_idx_pairs
        labels = []
        for criteria1_idx, criteria2_idx in pairs:
            labels.append(f"{self.criteria_vector.criteria_strs[criteria1_idx]}_{self.criteria_vector.criteria_strs[criteria2_idx]}")
        return labels
    
    def _makeDataframe(self, imat:int)->pd.DataFrame:
        """
        Make a dataframe of the matrix values.
        """
        values = self.values[imat, :, :]
        column_labels = self._makeColumnLabels()
        df = pd.DataFrame(values, columns=column_labels)
        return df

    def _makePairCriteriaCountMatrix(self, array:np.ndarray, criteria_vec:CriteriaVector)->Tuple[np.ndarray, list]:
        """
        Constructs 3-d matrix that describes pairwise comparisons of rows and criteria. Let m_i,j be the value of the
        stoichiometry matrix at row i and column j. Consider the pair of row i, ip and the pair of criteria k, kp.
        We want to calculate
            e_i,ip,k,kp = sum_n c_k(m_i,n) AND c_kp(m_ip,n)
        We structure this as a 3-d array:
            dim 1: i
            dim 2: ip
            dim 3: k,kp

        Args:
            array (np.array): An array of real numbers.
        Returns:
            criteria_count_arr (np.array): A 3-d array of counts of criteria pairs as described above.
            pairs of indices of criteria_vector
        """
        num_row = array.shape[0]
        num_criteria = criteria_vec.num_criteria
        # criteria_arr is a 3-d array of criteria values: criteria, row, column
        criteria_arr = np.array([c(array) for c in criteria_vec.criteria_functions])
        row_pairs = list(itertools.product(range(num_row), range(num_row)))
        criteria_pairs = list(itertools.product(range(num_criteria), range(num_criteria)))
        criteria1_idxs, criteria2_idxs = list(zip(*criteria_pairs))
        pair_count_arr = np.zeros((num_row, num_row, num_criteria*num_criteria), dtype=int)
        for row1, row2 in row_pairs:
            bools = criteria_arr[criteria1_idxs, row1, :] & criteria_arr[criteria2_idxs, row2, :]
            pair_count_arr[row1, row2, :] = np.sum(bools, axis=1)
        return pair_count_arr, criteria_pairs
    
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
        assignment = np.array(range(self.num_row))
        assignment = assignment[:assignment_len]
        assignment = np.reshape(assignment, (1, assignment_len))
        self._reference_matrix = self.getTargetArray(assignment)
        return self._reference_matrix

    def getTargetArray(self, assignments:np.ndarray[np.ndarray[int]])->np.ndarray:
        """
        Get a matrix as specified by the assignment. self.values is a 3-d array:
            1. Matrix selected
            2. Row in matrix
            3. Column in matrix

        Args:
            other_assignment (np.ndarray): Sequence of rows in original matrix that are to be compared

        Returns:
            np.ndarray: Two dimensional array in which rows being are flattened and the columns remain
        """
        idx1s = assignments[:, :-1].flatten()
        idx2s = assignments[:, 1:].flatten()
        return self.values[idx1s, idx2s, :]