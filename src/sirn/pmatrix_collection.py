'''Analysis of a collection of pmatrix.'''
"""
Structures the rows and columns of a matrix based on order independent properties of the rows and columns.
"""


from sirn.pmatrix import PMatrix   # type: ignore
from sirn.array_collection import ArrayCollection  # type: ignore

import numpy as np
from typing import List, Union, Optional


####################################
class PMatrixCollection(object):
        
    def __init__(self, pmatrices: List[PMatrix], is_permutably_identical:bool=False)->None:
        self.pmatrices = pmatrices
        self.is_permutably_identical = is_permutably_identical

    def __len__(self)->int:
        return len(self.pmatrices)
    
    def __repr__(self)->str:
        names = [p.model_name for p in self.pmatrices]
        return "---".join(names)
    
    @property
    def num_column(self)->Union[int, None]:
        if not self.is_permutably_identical:
            return None
        return self.pmatrices[0].num_column
    
    @property
    def hash_val(self)->Union[int, None]:
        if not self.is_permutably_identical:
            return None
        return self.pmatrices[0].hash_val
    
    @property
    def num_row(self)->Union[int, None]:
        if not self.is_permutably_identical:
            return None
        return self.pmatrices[0].num_row
    
    @property
    def row_collection(self)->Union[None, ArrayCollection]:
        if not self.is_permutably_identical:
            return None
        return self.pmatrices[0].row_collection
    
    @property
    def column_collection(self)->Union[None, ArrayCollection]:
        if not self.is_permutably_identical:
            return None
        return self.pmatrices[0].column_collection
    
    @classmethod
    def makeRandomCollection(cls, matrix_size:int=3, num_pmatrix:int=10)->'PMatrixCollection':
        """Make a random collection of pmatrices."""
        pmatrices = [PMatrix(np.random.randint(-1, 2, (matrix_size, matrix_size)))
                       for _ in range(num_pmatrix)]
        return cls(pmatrices)
    
    #def cluster(self)->List['PMatrixCollection']:
    def cluster(self):
        """
        Clusters the pmatrix in the collection by finding those that are permutably identical.

        Returns:
            List[PMatrixCollection]: A list of pmatrix collections. 
                    Each collection contains pmatrix that are permutably identical.
        """
        hash_dct = {}  # dict values are lists of pmatrix with the same hash value
        pmatrix_collections = []
        # Assign PMAtrix to hash_dct
        for pmatrix in self.pmatrices:
            if pmatrix.hash_val in hash_dct:
                hash_dct[pmatrix.hash_val].append(pmatrix)
            else:
                hash_dct[pmatrix.hash_val] = [pmatrix]
        # Construct the collections of permutably identical matrices
        for hash_val, pmatrices in hash_dct.items():
            new_collections = [pmatrices[0]]
            for pmatrix in pmatrices[1:]:
                is_permutably_identical = False
                for new_collection in new_collections:
                    if new_collection[0].isPermutablyIdentical(pmatrix):
                        new_collection.append(pmatrix)
                        is_permutably_identical = True
                        break
                if not is_permutably_identical:
                    new_collections.append([pmatrix])
            new_pmatrix_collections = [PMatrixCollection(pmatrices, is_permutably_identical=True) 
                                for pmatrices in new_collections]
            pmatrix_collections.extend(new_pmatrix_collections)
        return pmatrix_collections