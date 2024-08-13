'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from sirn import constants as cn  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.network_base import NetworkBase  # type: ignore

import collections
import numpy as np
from typing import Optional

CompatibilitySet = collections.namedtuple("CompatibilitySet", ["pos", "set"])
NULL_COMPATIBILITY_VECTOR:np.ndarray = np.array([])


class Network(NetworkBase):

    def __init__(self, reactant_arr:Matrix, 
                 product_arr:np.ndarray,
                 reaction_names:Optional[np.ndarray[str]]=None,
                 species_names:Optional[np.ndarray[str]]=None,
                 network_name:Optional[str]=None,
                 criteria_vector:Optional[CriteriaVector]=None)->None:
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            reaction_names (np.ndarray[str]): Names of the reactions.
            species_names (np.ndarray[str]): Names of the species
        """
        super().__init__(reactant_arr, product_arr, network_name=network_name,
                            reaction_names=reaction_names, species_names=species_names,
                            criteria_vector=criteria_vector) 

    def __eq__(self, other)->bool:
        """
        Args:
            other (Network): Network to compare to.
        Returns:
            bool: True if equal.
        """
        if not isinstance(other, Network):
            return False
        return super().__eq__(other)
    
    def copy(self):
        """
        Returns:
            Network: Copy of this network.
        """
        return Network(self.reactant_mat.values.copy(), self.product_mat.values.copy(),
                        network_name=self.network_name,
                        reaction_names=self.reaction_names,
                        species_names=self.species_names,
                        criteria_vector=self.criteria_vector)
    
    def makeCompatibilitySetVector(self,
          target:'Network',
          orientation:str,
          identity:str=cn.ID_WEAK,
          participant:Optional[str]=None,
          is_subsets:bool=False,
          )->np.ndarray[CompatibilitySet]:
        """
        Constructs a vector of lists of rows in the target that are compatible with each row in the reference (self).

        Args:
            target_network (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.

        Returns:
            np.ndarray: Vector of compatibility sets.
        """
        def makeBigArray(matrix:Matrix, other_num_row:int, is_block_repeats=False)->np.ndarray:
            """
            Constructs a large array that allows for simultaneous comparison of all rows.
              is_rotate: True: rotate the rows in the array
                         False: block repetitions

            Args:
                matrix (Matrix): Matrix to expand.
                other_num_row (int): Number of rows in other matrix.
                is_block_repeats (bool): If True, block repeats.

            Returns:
                np.ndarray: Expanded array.
            """
            # Convert to a linear array
            this_num_row, num_column = matrix.values.shape
            linear_arr = matrix.values.flatten()
            if is_block_repeats:
                arr = np.repeat(matrix.values, other_num_row, axis=0)
            else:
                repeat_arr = np.repeat(linear_arr, other_num_row, axis=0)
                arr = np.reshape(repeat_arr, (len(linear_arr), other_num_row)).T
                arr = np.reshape(arr, (this_num_row*other_num_row, num_column))
            return arr
        #
        if orientation == cn.OR_REACTION:
            this_num_row = self.num_reaction
        else:
            this_num_row = self.num_species
        compatible_sets:list = [ [] for _ in range(this_num_row)]
        if (not is_subsets) and not self.isStructurallyCompatible(target):
            pass
        elif is_subsets and not self.isSubsetCompatible(target):
            pass
        else:
            # Find compatible Rows
            if identity == cn.ID_WEAK:
                participant = None
            else:
                participant = participant
            reference_matrix = self.getNetworkMatrix(matrix_type=cn.MT_SINGLE_CRITERIA,
                                                      orientation=orientation,
                                                      identity=identity,
                                                      participant=participant)
            target_matrix = target.getNetworkMatrix(matrix_type=cn.MT_SINGLE_CRITERIA,
                                                      orientation=orientation,
                                                      identity=identity,
                                                      participant=participant)
            num_criteria = reference_matrix.num_column
            reference_num_row = reference_matrix.num_row
            target_num_row = target_matrix.num_row
            # Construct two large 2D arrays that allows for simultaneous comparison of all rows 
            big_reference_arr = makeBigArray(reference_matrix, target_num_row, is_block_repeats=True)
            big_target_arr = makeBigArray(target_matrix, reference_num_row, is_block_repeats=False)
            # Find the compatible sets
            if is_subsets:
                big_compatible_arr = big_reference_arr <= big_target_arr
            else:
                big_compatible_arr = big_reference_arr == big_target_arr
            satisfy_arr = np.sum(big_compatible_arr, axis=1) == num_criteria
            # Construct the sets
            target_indices = np.array(range(target_num_row))
            for iset in range(reference_num_row):
                indices = target_num_row*iset + target_indices
                compatible_sets[iset] = target_indices[satisfy_arr[indices]].tolist()
        #
        return np.array(compatible_sets)