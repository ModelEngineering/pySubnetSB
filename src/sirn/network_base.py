'''Container of properties for a reaction network.'''

"""
"""

from sirn import constants as cn  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn.util import hashArray  # type: ignore

import collections
import itertools
import os
import numpy as np
from typing import Optional, Union, List, Tuple


class NetworkBase(object):
    """
    Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.
    """

    def __init__(self, reactant_arr:Matrix, 
                 product_arr:np.ndarray,
                 reaction_names:Optional[List[str]]=None,
                 species_names:Optional[List[str]]=None,
                 network_name:Optional[str]=None,
                 criteria_vector:Optional[CriteriaVector]=None)->None:
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            reaction_names (List[str]): Names of the reactions.
            species_names (List[str]): Names of the species
        """
        # Reactant stoichiometry matrix is negative
        self.species_names = species_names
        self.reaction_names = reaction_names
        self.criteria_vec = criteria_vector
        self.reactant_mat = NamedMatrix(reactant_arr, row_names=species_names, column_names=reaction_names,
                                        row_description="species", column_description="reactions")
        self.product_mat = NamedMatrix(product_arr, row_names=species_names, column_names=reaction_names,
                                        row_description="species", column_description="reactions")
        # The following are deferred execution for efficiency considerations
        self._network_name = network_name
        self._stoichiometry_mat:Optional[NamedMatrix] = None
        self._network_mats:Optional[dict] = None # Network matrices populated on demand by getNetworkMat
        self._strong_hash:Optional[int] = None  # Hash for strong identity
        self._weak_hash:Optional[int] = None  # Hash for weak identity

    @property
    def weak_hash(self)->int:
        if self._weak_hash is None:
            stoichiometries = [self.getNetworkMatrix(
                                  matrix_type=cn.MT_SINGLE_CRITERIA, orientation=o,
                                  identity=cn.ID_WEAK)
                                  for o in [cn.OR_SPECIES, cn.OR_REACTION]]
            hash_arr = np.array([hashArray(stoichiometry.row_hashes) for stoichiometry in stoichiometries])
            self._weak_hash = hashArray(hash_arr)
        return self._weak_hash
        
    @property
    def strong_hash(self)->int:
        if self._strong_hash is None:
            stoichiometries:list = []
            for i_orientation in cn.OR_LST:
                for i_participant in cn.PR_LST:
                    stoichiometries.append(self.getNetworkMatrix(
                        matrix_type=cn.MT_SINGLE_CRITERIA,
                        orientation=i_orientation,
                        identity=cn.ID_STRONG,
                        participant=i_participant))
            hash_arr = np.array([hashArray(stoichiometry.row_hashes) for stoichiometry in stoichiometries])
            self._strong_hash = hashArray(hash_arr)
        return self._strong_hash

    @property
    def network_name(self)->str:
        if self._network_name is None:
            self._network_name = str(np.random.randint(0, 10000000))
        return self._network_name
    
    @property
    def stoichiometry_mat(self)->NamedMatrix:
        if self._stoichiometry_mat is None:
            stoichiometry_arr = self.product_mat.values - self.reactant_mat.values
            self._stoichiometry_mat = NamedMatrix(stoichiometry_arr, row_names=self.species_names,
               column_names=self.reaction_names, row_description="species", column_description="reactions")
        return self._stoichiometry_mat
    
    def getNetworkMatrix(self,
                         matrix_type:Optional[str]=None,
                         orientation:Optional[str]=None,
                         participant:Optional[str]=None,
                         identity:Optional[str]=None)->NamedMatrix:
        """
        Retrieves, possibly constructing, the matrix. The specific matrix is determined by the arguments.

        Args:
            marix_type: cn.MT_STANDARD, cn.MT_SINGLE_CRITERIA, cn.MT_PAIR_CRITERIA
            orientation: cn.OR_REACTION, cn.OR_SPECIES
            participant: cn.PR_REACTANT, cn.PR_PRODUCT
            identity: cn.ID_WEAK, cn.ID_STRONG

        Returns:
            subclass of Matrix
        """
        # Initialize the dictionary of matrices
        if self._network_mats is None:
            self._network_mats = {}
            for i_matrix_type in cn.MT_LST:
                for i_orientation in cn.OR_LST:
                    for i_identity in cn.ID_LST:
                        for i_participant in cn.PR_LST:
                            if i_identity == cn.ID_WEAK:
                                self._network_mats[(i_matrix_type, i_orientation, None, i_identity)] = None
                            else:
                                self._network_mats[(i_matrix_type, i_orientation, i_participant, i_identity)] = None
        # Check if the matrix is already in the dictionary
        if self._network_mats[(matrix_type, orientation, participant, identity)] is not None:
            return self._network_mats[(matrix_type, orientation, participant, identity)]
        # Obtain the matrix value
        #   Identity and participant
        if identity == cn.ID_WEAK:
            matrix = self.stoichiometry_mat
        elif identity == cn.ID_STRONG:
            if participant == cn.PR_REACTANT:
                matrix = self.reactant_mat
            elif participant == cn.PR_PRODUCT:
                matrix = self.product_mat
            else:
                raise ValueError("Invalid participant: {participant}.")
        else:
            raise ValueError("Invalid identity: {identity}.")
        #   Orientation
        if orientation == cn.OR_REACTION:
            matrix = matrix.transpose()
        elif orientation == cn.OR_SPECIES:
            pass
        else:
            raise ValueError("Invalid orientation: {orientation}.")
        #   Matrix type
        if matrix_type == cn.MT_SINGLE_CRITERIA:
            matrix = SingleCriteriaCountMatrix(matrix.values, criteria_vector=self.criteria_vec)
        elif matrix_type == cn.MT_PAIR_CRITERIA:
            matrix = PairCriteriaCountMatrix(matrix.values, criteria_vector=self.criteria_vec)
        elif matrix_type == cn.MT_STANDARD:
            pass
        else:
            raise ValueError("Invalid matrix type: {matrix_type}.")
        # Update the matrix
        self._network_mats[(matrix_type, orientation, participant, identity)] = matrix
        return matrix

    def copy(self)->'NetworkBase':
        return NetworkBase(self.reactant_mat.values.copy(), self.product_mat.values.copy(),
                       network_name=self.network_name, reaction_names=self.reaction_names,
                       species_names=self.species_names,
                       criteria_vector=self.criteria_vec)  # type: ignore

    def __repr__(self)->str:
        return self.network_name
    
    def __eq__(self, other)->bool:
        if self.network_name != other.network_name:
            return False
        if self.reactant_mat != other.reactant_mat:
            return False
        if self.product_mat != other.product_mat:
            return False
        return True
    
#    def randomize(self, structural_identity_type:str=cn.STRUCTURAL_IDENTITY_TYPE_STRONG,
#                  num_iteration:int=10, is_verify=True)->'Network':
#        """
#        Creates a new network with randomly permuted reactant and product matrices.
#
#        Args:
#            collection_identity_type (str): Type of identity collection
#            num_iteration (int): Number of iterations to find a randomized network
#            is_verify (bool): Verify that the network is structurally identical
#
#        Returns:
#            Network
#        """
#        is_found = False
#        for _ in range(num_iteration):
#            randomize_result = self.reactant_mat.randomize()
#            reactant_arr = randomize_result.pmatrix.array.copy()
#            if structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_STRONG:
#                is_structural_identity_type_weak = False
#                product_arr = PMatrix.permuteArray(self.product_mat.array,
#                        randomize_result.row_perm, randomize_result.column_perm) 
#            elif structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_WEAK:
#                is_structural_identity_type_weak = True
#                stoichiometry_arr = PMatrix.permuteArray(self.stoichiometry_mat.array,
#                        randomize_result.row_perm, randomize_result.column_perm) 
#                product_arr = reactant_arr + stoichiometry_arr
#            else:
#                # No requirement for being structurally identical
#                is_structural_identity_type_weak = True
#                randomize_result = self.product_mat.randomize()
#                product_arr = randomize_result.pmatrix.array
#            network = Network(reactant_arr, product_arr)
#            if not is_verify:
#                is_found = True
#                break
#            result =self.isStructurallyIdentical(network,
#                     is_structural_identity_weak=is_structural_identity_type_weak)
#            if (structural_identity_type==cn.STRUCTURAL_IDENTITY_TYPE_NOT):
#                is_found = True
#                break
#            elif (is_structural_identity_type_weak) and result.is_structural_identity_weak:
#                is_found = True
#                break
#            elif (not is_structural_identity_type_weak) and result.is_structural_identity_strong:
#                is_found = True
#                break
#            else:
#                pass
#        if not is_found:
#            raise ValueError("Could not find a randomized network. Try increasing num_iteration.")
#        return network
    
    @classmethod
    def makeFromAntimonyStr(cls, antimony_str:str, network_name:Optional[str]=None)->'NetworkBase':
        """
        Make a Network from an Antimony string.

        Args:
            antimony_str (str): Antimony string.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        stoichiometry = Stoichiometry(antimony_str)
        network = cls(stoichiometry.reactant_mat, stoichiometry.product_mat, network_name=network_name,
                      species_names=stoichiometry.species_names, reaction_names=stoichiometry.reaction_names)
        return network
                   
    @classmethod
    def makeFromAntimonyFile(cls, antimony_path:str,
                         network_name:Optional[str]=None)->'NetworkBase':
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
    def makeRandomNetwork(cls, species_array_size:int=5, reaction_array_size:int=5)->'NetworkBase':
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
        return NetworkBase(reactant_mat, product_mat)