'''Represents a vector of criteria that is a partition of the real line.'''

"""
For an input of N boundary values, the functions appear in the vector in the following order:
* 0 to N-1: Equality with boundary values
* N to 2N-2: Between boundary values
* 2N-1: Less than the first boundary value
* 2N: Greater than the last boundary value
"""

from sirn.matrix import Matrix # type: ignore
import sirn.util as util # type: ignore
import sirn.constants as cn # type: ignore

import numpy as np
import pickle
from typing import List, Union


class CriteriaVector(object):
    # Creates a vector of criteria that is a partition of the real line. Criteria are functions that test
    # for equality with a boundary or being between boundary values.
    # The partitions are: (a) equality for boundary values and (b) all other values in one categor
    def __init__(self, boundary_values: List[float]=cn.CRITERIA_BOUNDARY_VALUES):
        """
        Args:
            criteria (np.array): A vector of criteria.
        """
        self.boundary_values = boundary_values
        self.criteria_functions, self.criteria_strs = self._makeCriteria()
        self.num_criteria = len(self.criteria_functions)

    def copy(self):
        return CriteriaVector(self.boundary_values)
    
    def serialize(self)->util.ArrayContext:
        # Creates a pickle string for the input object
        return util.array2Context(self.boundary_values)

    def _makeCriteria(self):
        """"
        Returns:
            np.array: A vector of criteria
            list: A list of strings describing the criteria
        """
        criteria = []
        criteria_strs = []   # Strings describing the functions
        # Construct criteria for equality with boundary values
        for val in self.boundary_values:
            idx = len(criteria)
            function_name = f'function_{idx}'
            exec(f'def {function_name}(x):\n    return x == {val}')
            criteria.append(locals()[function_name])
            criteria_strs.append(f'={val}')
        # Catch anything else
        idx += 1
        function_name = f'function_{idx}'
        repeat_statement = " & ".join([f'(x != {v})' for v in self.boundary_values])
        statement = f'def {function_name}(x):\n    return {repeat_statement}'
        exec(statement)
        criteria.append(locals()[function_name])
        criteria_strs.append(f'!=others')
        #
        return criteria, criteria_strs