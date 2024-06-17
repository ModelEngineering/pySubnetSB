'''Analysis of a collection of pmatrix.'''
"""
Structures the rows and columns of a matrix based on order independent properties of the rows and columns.

A key operation on a collection is cluster. 
The cluster operation groups constructs a collection of pmatrix_collection
each of which contains permutably identical matrices.
"""


from sirn.pmatrix import PMatrix   # type: ignore
from sirn.array_collection import ArrayCollection  # type: ignore

import numpy as np
from typing import List, Union, Dict


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
    
    def __add__(self, other:'PMatrixCollection')->'PMatrixCollection':
        pmatrix_collection = self.copy()
        is_permutably_identical = self.is_permutably_identical and other.is_permutably_identical
        if is_permutably_identical:
            is_permutably_identical = self.pmatrices[0].isPermutablyIdentical(other.pmatrices[0])
        pmatrix_collection.is_permutably_identical = is_permutably_identical
        pmatrix_collection.pmatrices.extend(other.pmatrices)
        return pmatrix_collection
    
    def copy(self)->'PMatrixCollection':
        return PMatrixCollection(self.pmatrices.copy(),
                                 is_permutably_identical=self.is_permutably_identical)
    
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
    
    def cluster(self, is_report=True)->List['PMatrixCollection']:
        """
        Clusters the pmatrix in the collection by finding those that are permutably identical.

        Returns:
            List[PMatrixCollection]: A list of pmatrix collections. 
                    Each collection contains pmatrix that are permutably identical.
        """
        hash_dct: Dict[int, List[PMatrix]] = {}  # dict values are lists of pmatrix with the same hash value
        pmatrix_collections = []  # list of permutably identical pmatrix collections
        # Build the hash dictionary
        for pmatrix in self.pmatrices:
            if pmatrix.hash_val in hash_dct:
                hash_dct[pmatrix.hash_val].append(pmatrix)
            else:
                hash_dct[pmatrix.hash_val] = [pmatrix]
        if is_report:
            print(f"**Number of hash values: {len(hash_dct)}")
        # Construct the collections of permutably identical matrices
        for idx, pmatrices in enumerate(hash_dct.values()):  # Iterate over collections of pmatrice with the same hash value
            # Find collections of permutably identical matrices
            first_collection = [pmatrices[0]]
            new_collections = [first_collection]  # list of collections of permutably identical matrices
            for pmatrix in pmatrices[1:]:
                is_in_existing_collection = False
                for new_collection in new_collections:
                    if new_collection[0].isPermutablyIdentical(pmatrix):
                        new_collection.append(pmatrix)
                        is_in_existing_collection = True
                        break
                if not is_in_existing_collection:
                    new_collections.append([pmatrix])
            if is_report:
                print(".", end='')
            new_pmatrix_collections = [PMatrixCollection(pmatrices, is_permutably_identical=True) 
                                for pmatrices in new_collections]
            pmatrix_collections.extend(new_pmatrix_collections)
        return pmatrix_collections