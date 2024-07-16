'''Encoding of a one dimensional array'''

import sirn.constants as cn # type: ignore

import itertools
import numpy as np
import scipy  # type: ignore
import scipy.special  # type: ignore
from typing import Dict

VALUE_SEPARATOR = ','  # Separates the values in a string encoding


class Encoding(object):

    def __init__(self, array: np.ndarray)->None:
        """
        Args:
            arrays (np.array): A collection of arrays.
            is_weighted (bool): Weight the value of the i-th element in an array by the sum of non-zero
                elements in the i-th position.
        """
        self.array = array
        self.sorted_arr = np.sort(array)
        self.encoding_val = ",".join([str(x) for x in self.sorted_arr])
        self.value_arr = np.array(list(set(self.sorted_arr)))   # Array of distinct values
        self.count_dct = {x: np.sum(self.sorted_arr == x) for x in self.value_arr}

    def __repr__(self)->str:
        return self.encoding_val
    
    def __eq__(self, other )->bool:
        if not isinstance(other, Encoding):
            return False
        return self.encoding_val == other.encoding_val
    
    def __gt__(self, other)->bool:
        if not isinstance(other, Encoding):
            return False
        return self.encoding_val > other.encoding_val
    
    def isCompatibleSubset(self, other: 'Encoding')->bool:
        if not set(self.value_arr).issubset(set(other.value_arr)):
            return False
        return all([self.count_dct[x] <= other.count_dct[x] for x in self.value_arr])