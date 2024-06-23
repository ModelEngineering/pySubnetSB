'''Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.'''

from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn.pmatrix import PMatrix  # type: ignore
from sirn.util import hashArray  # type: ignore
from sirn import constants as cn  # type: ignore

import os
import numpy as np
from typing import Optional, Union


class Network(object):
    """
    Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.
    """

    def __init__(self, reactant_mat:Union[np.ndarray, PMatrix],
                 product_mat:Union[np.ndarray, PMatrix],
                 network_name:Optional[str]=None):
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
        """
        if isinstance(reactant_mat, PMatrix):
            self.reactant_pmatrix = reactant_mat
        else:
            self.reactant_pmatrix = PMatrix(reactant_mat)
        if isinstance(product_mat, PMatrix):
            self.product_pmatrix = product_mat
        else:
            self.product_pmatrix = PMatrix(product_mat)
        if network_name is None:
            network_name = str(np.random.randint(0, 100000))
        self.network_name = network_name
        stoichiometry_array = self.product_pmatrix.array - self.reactant_pmatrix.array
        self.stoichiometry_pmatrix = PMatrix(stoichiometry_array)
        # Hash values for simple stoichiometry (only stoichiometry matrix) and non-simple stoichiometry
        self.nonsimple_hash = hashArray(np.array([self.reactant_pmatrix.hash_val,
                                                  self.product_pmatrix.hash_val]))
        self.simple_hash = self.stoichiometry_pmatrix.hash_val

    def copy(self)->'Network':
        return Network(self.reactant_pmatrix.array.copy(), self.product_pmatrix.array.copy(),
                       network_name=self.network_name)

    def __repr__(self)->str:
        return self.network_name
    
    def __eq__(self, other)->bool:
        if self.network_name != other.network_name:
            return False
        if self.reactant_pmatrix != other.reactant_pmatrix:
            return False
        if self.product_pmatrix != other.product_pmatrix:
            return False
        return True
    
    def isStructurallyIdentical(self, other:'Network', is_structural_identity_type_weak:bool=False)->bool:
        """
        Determines if two networks are structurally identical. This means that the reactant and product
        matrices are identical, up to a permutation of the rows and columns. If tructural_identity_weak
        is True, then the stoichiometry matrix is also checked for a permutation that makes the product
        stoichiometry matrix equal to the reactant stoichiometry matrix.

        Args:
            other (Network): _description_
            is_structural_identity_type_weak (bool, optional): _description_. Defaults to False.

        Returns:
            bool
        """
        if is_structural_identity_type_weak:
            return bool(self.stoichiometry_pmatrix.isPermutablyIdentical(other.stoichiometry_pmatrix))
        # Check that the combined hash (reactant_pmatrix, product_pmatrix) is the same.
        if self.nonsimple_hash != other.nonsimple_hash:
            return False
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
    
    def randomize(self, structural_identity_type:str=cn.STRUCTURAL_IDENTITY_TYPE_STRONG,
                  num_iteration:int=10)->'Network':
        """
        Creates a new network with randomly permuted reactant and product matrices.

        Args:
            collection_identity_type (str): Type of identity collection
            num_iteration (int): Number of iterations to find a randomized network

        Returns:
            Network
        """
        is_found = False
        for _ in range(num_iteration):
            randomize_result = self.reactant_pmatrix.randomize()
            reactant_arr = randomize_result.pmatrix.array.copy()
            is_structural_identity_type_weak = None
            if structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_STRONG:
                is_structural_identity_type_weak = False
                product_arr = PMatrix.permuteArray(self.product_pmatrix.array,
                        randomize_result.row_perm, randomize_result.column_perm) 
            elif structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_WEAK:
                is_structural_identity_type_weak = True
                stoichiometry_arr = PMatrix.permuteArray(self.stoichiometry_pmatrix.array,
                        randomize_result.row_perm, randomize_result.column_perm) 
                product_arr = reactant_arr + stoichiometry_arr
            else:
                # No requirement for being structurally identical
                randomize_result = self.product_pmatrix.randomize()
                product_arr = randomize_result.pmatrix.array
            network = Network(reactant_arr, product_arr)
            if is_structural_identity_type_weak is None:
                # Cannot be weakly structurally identical
                if not self.isStructurallyIdentical(network, is_structural_identity_type_weak=True):
                    is_found = True
                    break
            else:
                if self.isStructurallyIdentical(network,
                        is_structural_identity_type_weak=is_structural_identity_type_weak):
                    is_found = True
                    break
        if not is_found:
            raise ValueError("Could not find a randomized network. Try increasing num_iteration.")
        return network
    
    @classmethod
    def makeFromAntimonyStr(cls, antimony_str:str, network_name:Optional[str]=None)->'Network':
        """
        Make a Network from an Antimony string.

        Args:
            antimony_str (str): Antimony string.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        stoichiometry = Stoichiometry(antimony_str)
        reactant_pmatrix = PMatrix(stoichiometry.reactant_mat, row_names=stoichiometry.species_names,
                                   column_names=stoichiometry.reaction_names)
        product_pmatrix = PMatrix(stoichiometry.product_mat, row_names=stoichiometry.species_names,
                                   column_names=stoichiometry.reaction_names)
        network = cls(reactant_pmatrix, product_pmatrix, network_name=network_name)
        return network
                   
                   
    
    @classmethod
    def makeFromAntimonyFile(cls, antimony_path:str, is_structurally_identical:bool=False,
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
        return cls.makeFromAntimonyStr(antimony_str, network_name=network_name)