'''Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.'''

from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn.pmatrix import PMatrix  # type: ignore
from sirn.util import hashArray  # type: ignore
from sirn import constants as cn  # type: ignore

import os
import numpy as np
from typing import Optional, Union


class StructurallyIdenticalResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
                 is_compatible:bool=False,
                 is_structural_identity_weak:bool=False,
                 is_structural_identity_strong:bool=False,
                 is_excessive_perm:bool=False,
                 num_perm:int=0,
                 ):
        """
        Args:
            is_compatible (bool): has the same shapes and row/column encodings
            is_structural_identity_weak (bool)
            is_structural_identity_strong (bool)
            is_excessive_perm (bool): the number of permutations exceeds a threshold
        """
        self.is_excessive_perm = is_excessive_perm
        self.is_compatible = is_compatible
        self.is_structural_identity_weak = is_structural_identity_weak
        self.is_structural_identity_strong = is_structural_identity_strong
        self.num_perm = num_perm
        #
        if self.is_structural_identity_strong:
            self.is_structural_identity_weak = True
        if self.is_structural_identity_weak:
            self.is_compatible = True
        if is_excessive_perm:
            self.is_structural_identity_weak = False
            self.is_structural_identity_strong = False


class Network(object):
    """
    Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.
    """

    def __init__(self, reactant_mat:Union[np.ndarray, PMatrix],
                 product_mat:Union[np.ndarray, PMatrix],
                 network_name:Optional[str]=None)->None:
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
            network_name = str(np.random.randint(0, 10000000))
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
    
    def isStructurallyIdentical(self, other:'Network', is_report:bool=False,
            max_num_perm:int=cn.MAX_NUM_PERM,
            is_structural_identity_weak:bool=False)->StructurallyIdenticalResult:
        """
        Determines if two networks are structurally identical. This means that the reactant and product
        matrices are identical, up to a permutation of the rows and columns. If tructural_identity_weak
        is True, then the stoichiometry matrix is also checked for a permutation that makes the product
        stoichiometry matrix equal to the reactant stoichiometry matrix.

        Args:
            other (Network)
            max_num_perm (int, optional): Maximum number of permutations to consider.
            is_report (bool, optional): Report on analysis progress. Defaults to False.
            is_structural_identity_type_weak (bool, optional): _description_. Defaults to False.

        Returns:
            StructurallyIdenticalResult
        """
        num_perm = 0
        # Check for weak structural identity
        weak_identity = self.stoichiometry_pmatrix.isPermutablyIdentical(
            other.stoichiometry_pmatrix, is_find_all_perms=False, max_num_perm=max_num_perm)
        num_perm += weak_identity.num_perm
        if num_perm >= max_num_perm:
            return StructurallyIdenticalResult(num_perm=num_perm, is_excessive_perm=True)
        if is_structural_identity_weak:
            return StructurallyIdenticalResult(is_compatible=weak_identity.is_compatible,
                    is_structural_identity_weak=weak_identity.is_permutably_identical,
                    is_excessive_perm=weak_identity.is_excessive_perm,
                    num_perm=num_perm)
        # Check that the combined hash (reactant_pmatrix, product_pmatrix) is the same.
        if self.nonsimple_hash != other.nonsimple_hash:
            return StructurallyIdenticalResult(is_structural_identity_weak=True,
                                               num_perm=num_perm)
        # Find the permutations that work for weak identity and see if one works for strong identity
        revised_max_num_perm = max_num_perm - num_perm
        all_weak_identities = self.stoichiometry_pmatrix.isPermutablyIdentical(
            other.stoichiometry_pmatrix, is_find_all_perms=True, max_num_perm=revised_max_num_perm)
        num_perm += all_weak_identities.num_perm
        if num_perm >= max_num_perm:
            return StructurallyIdenticalResult(is_structural_identity_weak=True,
                    num_perm=num_perm, is_excessive_perm=True)
        if is_report:
            print(f'all_weak_identities: {len(all_weak_identities.this_column_perms)}')
        other_array = PMatrix.permuteArray(other.product_pmatrix.array,
                     all_weak_identities.other_row_perm,     # type: ignore
                     all_weak_identities.other_column_perm)  # type: ignore
        # Look at each permutation that makes the stoichiometry matrices equal
        for idx, this_row_perm in enumerate(all_weak_identities.this_row_perms):
            if num_perm >= max_num_perm:
                return StructurallyIdenticalResult(is_structural_identity_weak=True,
                                                   num_perm=num_perm,
                                                   is_excessive_perm=True)
            this_column_perm = all_weak_identities.this_column_perms[idx]
            this_array = PMatrix.permuteArray(self.product_pmatrix.array,
                                              this_row_perm, this_column_perm)
            num_perm += 1
            if bool(np.all([x == y for x, y in zip(this_array, other_array)])):
                return StructurallyIdenticalResult(is_structural_identity_strong=True,
                                                   num_perm=num_perm)
        # Has weak structurally identity but not strong
        return StructurallyIdenticalResult(is_structural_identity_weak=True,
                                           num_perm=num_perm)
    
    def randomize(self, structural_identity_type:str=cn.STRUCTURAL_IDENTITY_TYPE_STRONG,
                  num_iteration:int=10, is_verify=True)->'Network':
        """
        Creates a new network with randomly permuted reactant and product matrices.

        Args:
            collection_identity_type (str): Type of identity collection
            num_iteration (int): Number of iterations to find a randomized network
            is_verify (bool): Verify that the network is structurally identical

        Returns:
            Network
        """
        is_found = False
        for _ in range(num_iteration):
            randomize_result = self.reactant_pmatrix.randomize()
            reactant_arr = randomize_result.pmatrix.array.copy()
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
                is_structural_identity_type_weak = True
                randomize_result = self.product_pmatrix.randomize()
                product_arr = randomize_result.pmatrix.array
            network = Network(reactant_arr, product_arr)
            if not is_verify:
                is_found = True
                break
            result =self.isStructurallyIdentical(network,
                     is_structural_identity_weak=is_structural_identity_type_weak)
            if (structural_identity_type==cn.STRUCTURAL_IDENTITY_TYPE_NOT):
                is_found = True
                break
            elif (is_structural_identity_type_weak) and result.is_structural_identity_weak:
                is_found = True
                break
            elif (not is_structural_identity_type_weak) and result.is_structural_identity_strong:
                is_found = True
                break
            else:
                pass
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
    def makeFromAntimonyFile(cls, antimony_path:str,
                         network_name:Optional[str]=None)->'Network':
        """
        Make a Network from an Antimony file. The default network name is the file name.

        Args:
            antimony_path (str): path to an Antimony file.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        with open(antimony_path, 'r') as fd:
            antimony_str = fd.read()
        if network_name is None:
            filename = os.path.basename(antimony_path)
            network_name = filename.split('.')[0]
        return cls.makeFromAntimonyStr(antimony_str, network_name=network_name)
    
    @classmethod
    def makeRandomNetwork(cls, species_array_size:int=5, reaction_array_size:int=5)->'Network':
        """
        Makes a random network.

        Args:
            species_array_size (int): Number of species.
            reaction_array_size (int): Number of reactions.

        Returns:
            Network
        """
        reactant_mat = np.random.randint(-1, 2, (species_array_size, reaction_array_size))
        product_mat = np.random.randint(-1, 2, (species_array_size, reaction_array_size))
        return Network(reactant_mat, product_mat)