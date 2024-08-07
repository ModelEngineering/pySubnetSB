'''A PairCriteriaCountMatrix counts of co-occurrences of criteria pairs satisfied by pairs of rows.'''

from sirn import constants as cn # type: ignore
from sirn.criteria_vector import CriteriaVector # type: ignore
from sirn.criteria_count_matrix import CriteriaCountMatrix # type: ignore
from sirn.named_matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore

import itertools
import numpy as np
from typing import Tuple, Optional

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
        values = self._makePairCriteriaCountMatrix(array, criteria_vector)
        super().__init__(values, criteria_vector=criteria_vector)
    
    def index2Pair(self, index:int)->Tuple[int, int]:
        """
        Convert an index to a pair of row indices. Pairs are counted by row
        and then by column.
        Args:
            index (int): An index.
        Returns:
            Tuple[int, int]: A pair of row indices.
        """
        return divmod(index, self.num_column)
    
    def pair2Index(self, pair:Tuple[int, int])->int:
        """
        Convert a pair of row indices to an index. Pairs are counted by row
        and then by columns.
        Args:
            pair (Tuple[int, int]): A pair of row indices.
        Returns:
            int: An index.
        """
        return pair[0]*self.num_column + pair[1]

    def _makePairCriteriaCountMatrix(self, values:np.ndarray, criteria_vec:CriteriaVector)->Matrix:
        """
        Evaluate the criteria on an array.
        Args:
            array (np.array): An array of real numbers.
        Returns:
            np.array: A matrix of boolean values.
        """
        num_row, num_column = values.shape
        array = np.array([c(values.T) for c in criteria_vec])
        array = array.T
        column_pairs = itertools.combinations(range(num_column), 2)
        row_pairs = itertools.combinations(range(num_row), 2)
        # Construct repetitions of rows of matrix
        rows = []
        for row_pair in row_pairs:
            row = []
            for column_pair in column_pairs:
                row.append(np.logical_and(array[row_pair[0], column_pair[0]], array[row_pair[1], column_pair[1]]))
            rows.append(row)
        return Matrix(rows)