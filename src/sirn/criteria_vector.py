'''Represents a vector of criteria that is a partition of the real line.'''

"""
For an input of N boundary values, the functions appear in the vector in the following order:
* 0 to N-1: Equality with boundary values
* N to 2N-2: Between boundary values
* 2N-1: Less than the first boundary value
* 2N: Greater than the last boundary value
"""

from sirn.matrix import Matrix # type: ignore

import numpy as np
from typing import List, Union

BOUNDARY_VALUES = [-1.0, 0.0, 1.0]


class CriteriaVector(object):
    # Creates a vector of criteria that is a partition of the real line. Criteria are functions that test
    # for equality with a boundary or being between boundary values.
    def __init__(self, boundary_values: List[float]=BOUNDARY_VALUES):
        """
        Args:
            criteria (np.array): A vector of criteria.
        """
        self.boundary_values = boundary_values
        self.criteria_functions = self._makeCriteria()
        self.num_criteria = len(self.criteria_functions)

    def _makeCriteria(self):
        criteria = []
        # Construct criteria for equality with boundary values
        for val in self.boundary_values:
            idx = len(criteria)
            function_name = f'function_{idx}'
            exec(f'def {function_name}(x):\n    return x == {val}')
            criteria.append(locals()[function_name])
        # Construct criteria for being between boundary values
        for i in range(len(self.boundary_values) - 1):
            idx = len(criteria)
            function_name = f'function_{idx}'
            lower = self.boundary_values[i]
            upper = self.boundary_values[i+1]
            exec(f'def {function_name}(x):\n    return np.logical_and((x > {lower}), (x < {upper}))')
            criteria.append(locals()[function_name])
        # Construct criteria for endpoints
        idx = len(criteria)
        function_name = f'function_{idx}'
        exec(f'def {function_name}(x):\n    return x < {self.boundary_values[0]}')
        criteria.append(locals()[function_name])
        #
        idx = len(criteria)
        function_name = f'function_{idx}'
        exec(f'def {function_name}(x):\n    return x > {self.boundary_values[-1]}')
        criteria.append(locals()[function_name])
        #
        return criteria