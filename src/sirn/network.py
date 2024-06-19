'''Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.'''

from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn.pmatrix import PMatrix  # type: ignore

import os
import numpy as np
from typing import Optional


class Network(object):
    """
    Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.
    """

    def __init__(self, reactant_mat:np.ndarray, product_mat:np.ndarray,
                 network_name:Optional[str]=None, is_simple_stoichiometry:bool=False):
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            is_simple_stoichiometry (bool): If True, then test for identical structure
                is based of the difference between the product and reactant stoichiometry matrices.
        """
        self.reactant_pmatrix = PMatrix(reactant_mat)
        self.product_pmatrix = PMatrix(product_mat)
        if network_name is None:
            network_name = str(np.random.randint(0, 100000))
        self.network_name = network_name
        stoichiometry_array = self.product_pmatrix.array - self.reactant_pmatrix.array
        self.stoichiometry_pmatrix = PMatrix(stoichiometry_array)
        self.is_simple_stoichiometry = is_simple_stoichiometry

    def copy(self)->'Network':
        return Network(self.reactant_pmatrix.array.copy(), self.product_pmatrix.array.copy(),
                       network_name=self.network_name,
                       is_simple_stoichiometry=self.is_simple_stoichiometry)

    def __repr__(self)->str:
        return self.network_name
    
    def __eq__(self, other)->bool:
        if self.network_name != other.network_name:
            return False
        if self.reactant_pmatrix != other.reactant_pmatrix:
            return False
        if self.product_pmatrix != other.product_pmatrix:
            return False
        if self.is_simple_stoichiometry != other.is_simple_stoichiometry:
            return False
        return True
    
    def isStructurallyIdentical(self, other)->bool:
        if self.is_simple_stoichiometry:
            return bool(self.stoichiometry_pmatrix.isPermutablyIdentical(other.stoichiometry_pmatrix))
        # Must separately check the reactant and product matrices.
        result = self.reactant_pmatrix.isPermutablyIdentical(other.reactant_pmatrix)
        if not result:
            return False
        # Now check that there is a permutation of the reactant stoichiometry matrix
        # that makes the product stoichiometry matrices equal.
        other_array = PMatrix.permuteArray(other.product_pmatrix.array,
                     result.other_row_perm, result.other_column_perm)  # type: ignore
        for idx, this_row_perm in enumerate(result.this_row_perms):
            this_column_perm = result.this_column_perms[idx]
            this_array = PMatrix.permuteArray(self.product_pmatrix.array, this_row_perm, this_column_perm)
            if bool(np.all([x == y for x, y in zip(this_array, other_array)])):
                return True
        return False
    
    @classmethod
    def makeAntimony(cls, antimony_str:str, network_name:Optional[str]=None,
                     is_simple_stoichiometry=True)->'Network':
        """
        Make a Network from an Antimony string.

        Args:
            antimony_str (str): Antimony string.
            network_name (str): Name of the network.
            is_simple_stoichiometry (bool): If True, then test for identical structure

        Returns:
            Network
        """
        stoichiometry = Stoichiometry(antimony_str)
        return cls(stoichiometry.reactant_mat, stoichiometry.product_mat, network_name=network_name,
                   is_simple_stoichiometry=is_simple_stoichiometry)
    
    @classmethod
    def makeAntimonyFile(cls, antimony_path:str, is_structurally_identical:bool=False,
                         network_name:Optional[str]=None,
                         is_simple_stoichiometry:bool=False)->'Network':
        """
        Make a Network from an Antimony file. The default network name is the file name.

        Args:
            antimony_path (str): path to an Antimony file.
            network_name (str): Name of the network.
            is_structurally_identical (bool): If True, then test for identical structure
            is_simple_stoichiometry (bool): If True, then test for identical structure

        Returns:
            Network
        """
        with open(antimony_path, 'r') as fd:
            antimony_str = fd.read()
        if network_name is None:
            filename = os.path.basename(antimony_path)
            network_name = filename.split('.')[0]
        return cls.makeAntimony(antimony_str, network_name=network_name,
                                is_simple_stoichiometry=is_simple_stoichiometry)