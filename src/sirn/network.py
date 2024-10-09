'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from sirn import constants as cn  # type: ignore
from sirn import util  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.reaction_constraint import ReactionConstraint  # type: ignore
from sirn.species_constraint import SpeciesConstraint   # type: ignore
from sirn.network_base import NetworkBase, AssignmentPair  # type: ignore

import collections
import pynauty  # type: ignore
import numpy as np
from typing import Optional, List, Tuple, Union


IS_DEBUG = False

NULL_ARRAY = np.array([])  # Null array
ATTRS = ["reactant_mat", "product_mat", "reaction_names", "species_names", "network_name"]
MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis


class StructurallyIdenticalResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
                 assignment_pairs:list[AssignmentPair],
                 is_truncated:Optional[bool]=False,
                 )->None:
        """
        Args:
            assignment_pairs (list[AssignmentPair]): List of assignment pairs.
            is_trucnated (bool): True if the number of assignments exceeds the maximum number of assignments.
        """
        self.assignment_pairs = assignment_pairs
        self.is_truncated = is_truncated

    def __bool__(self)->bool:
        return len(self.assignment_pairs) > 0
    
    def __repr__(self)->str:
        repr = f"StructurallyIdenticalResult(assignment_pairs={self.assignment_pairs};"
        repr += f" is_truncated={self.is_truncated};"
        return repr


class Network(NetworkBase):

    def __init__(self, reactant_arr:Union[np.ndarray, Matrix], 
                 product_arr:Union[np.ndarray, Matrix],
                 reaction_names:Optional[np.ndarray[str]]=None, # type: ignore
                 species_names:Optional[np.ndarray[str]]=None,  # type: ignore
                 network_name:Optional[str]=None)->None:               # type: ignore
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            reaction_names (np.ndarray[str]): Names of the reactions.
            species_names (np.ndarray[str]): Names of the species
        """
        if isinstance(reactant_arr, Matrix):
            reactant_arr = reactant_arr.values
        if isinstance(product_arr, Matrix):
            product_arr = product_arr.values
        super().__init__(reactant_arr, product_arr, network_name=network_name,
                            reaction_names=reaction_names, species_names=species_names)
        
    def isEquivalent(self, other)->bool:
        """Same except for the network name.

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if not isinstance(other, self.__class__):
            return False
        return super().isEquivalent(other)

    def __eq__(self, other)->bool:
        """
        Args:
            other (Network): Network to compare to.
        Returns:
            bool: True if equal.
        """
        if not isinstance(other, self.__class__):
            return False
        return super().__eq__(other)
    
    def copy(self):
        """
        Returns:
            Network: Copy of this network.
        """
        return Network(self.reactant_nmat.values.copy(), self.product_nmat.values.copy(),
                        network_name=self.network_name,
                        reaction_names=self.reaction_names,
                        species_names=self.species_names,
                        criteria_vector=self.criteria_vector)
    
    def isIsomorphic(self, target:'Network')->bool:
        """Using pynauty to detect isomorphism of reaction networks.

        Args:
            target (Network)

        Returns:
            bool
        """
        self_graph = self.makePynautyNetwork()
        target_graph = target.makePynautyNetwork()
        return pynauty.isomorphic(self_graph, target_graph)

    @staticmethod 
    def _compare(reference_arr:np.ndarray, target_arr:np.ndarray,
          species_assignment_arr:np.ndarray, reaction_assignment_arr:np.ndarray)->np.ndarray[bool]:  # type: ignore
            """
            Compares the reference matrix to the target matrix using a vectorized approach. Checks if the
            reference reactions have the same number of species as their corresponding target reaction.

            Args:
                reference_arr: np.ndarray - reference stoichiometry matrix
                target_arr: np.ndarray  - target stoichiometry matrix
                species_assignment_arr: np.ndarray (assignment of target to reference)
                reaction_assignment_arr: np.ndarray (assignment of target to reference)

            Returns:
                np.ndarray[bool]: flattened boolean array of the outcome of comparisons for pairs of assignments
                                  (i.e., assignments of target species to reference species
                                  and target reactions to reference reactions). The n-th entry in the array corresponds
                                  to the results of comparing the i-th species assignment and j-th reaction assignment
                                  with the target, where n = i + j*num_species_assignment.
            """
            #print("species_assignment", species_assignment_arr.shape)
            #print("reaction_assignment", reaction_assignment_arr.shape)
            # Size the comparison arrays used to make comparisons
            num_species_assignment, num_reference_species =  species_assignment_arr.shape
            num_reaction_assignment, num_reference_reaction =  reaction_assignment_arr.shape
            num_assignment =  num_species_assignment*num_reaction_assignment
            # Set up the comparison arrays. These are referred to as 'big' arrays.
            big_reference_arr = np.concatenate([reference_arr]*num_assignment, axis=0)
            #print("big_reference_arr", np.shape(big_reference_arr))
            big_reference_arr = big_reference_arr.astype(int)
            species_idxs = species_assignment_arr.flatten()
            #reaction_idxs = reaction_assignment_arr.flatten()
            # The following constructs the re-arranged target matrix.
            #   This is organized as blocks of row rearrangements (species) and
            #   each row rearrangement is repeated for several columns (reactions) rearrangements.
            #   Indices are pair to select each row and column in the target matrix
            # 
            #   Create the sequence of reaction indices that is paired with the sequence of species indices
            #   This repeat the reaction indices for each species (row), and does this for each
            #   reaction assignment.
            #structured_reaction_idxs = np.reshape(reaction_idxs, (num_reaction_assignment, num_reference_reaction))
            big_structured_reaction1_idxs = np.concatenate([reaction_assignment_arr]*num_reference_species,
                  axis=1)
            #print("big_structured_reaction1_idxs", np.shape(big_structured_reaction1_idxs))
            big_structured_reaction2_idxs = np.concatenate([big_structured_reaction1_idxs]*num_species_assignment,
                  axis=0)
            #print("big_structured_reaction2_idxs", np.shape(big_structured_reaction2_idxs))
            big_reaction_idxs = big_structured_reaction2_idxs.flatten()
            #  Create the sequence of species indices that is paired with the sequence of reaction indices
            big_species1_idxs = np.repeat(species_idxs, num_reference_reaction)
            #  Reshape to the shape of the stoichiometry matrix
            big_species2_idxs = np.reshape(big_species1_idxs,
                  (num_species_assignment, num_reference_species*num_reference_reaction))
            #  Replicate for each reaction assignment
            big_species3_idxs = np.concatenate([big_species2_idxs]*num_reaction_assignment,
                  axis=1)
            #  Convert to a single array
            big_species_idxs = big_species3_idxs.flatten()
            #   The following constructs the re-arranged target matrix.
            big_target_arr = np.reshape(target_arr[big_species_idxs, big_reaction_idxs],
                  (num_assignment*num_reference_species, num_reference_reaction)).astype(int)
            # Compare each row
            big_compatible_arr = big_reference_arr == big_target_arr
            big_row_sum = np.sum(big_compatible_arr, axis=1)
            big_row_satisfy = big_row_sum == num_reference_reaction
            # Rows are results of the comparison of the reference and target; columns are assignments
            assignment_pair_satisfy_arr = np.reshape(big_row_satisfy, (num_assignment, num_reference_species))
            # Index is True if the assignment-pair results in an identical matrix
            assignment_satisfy_arr = np.sum(assignment_pair_satisfy_arr, axis=1) == num_reference_species
            if IS_DEBUG:
                if np.sum(assignment_satisfy_arr) == 0:
                    import pdb; pdb.set_trace()
            return assignment_satisfy_arr  # Booleans indicating acceptable assignments
    
    def isStructurallyIdentical(self, target:'Network', is_subset:bool=False,
            max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT, identity:str=cn.ID_WEAK,
            expected_assignment_pair:Optional[AssignmentPair]=None)->StructurallyIdenticalResult:
        """
        Determines if the network is structurally identical to another network or subnet of another network.

        Args:
            target (Network): Network to search for structurally identity
            is_subsets (bool, optional): Consider subsets
            max_num_assignment (int, optional): Maximum number of assignments to produce.
            identity (str, optional): cn.ID_WEAK or cn.ID_STRONG
            expected_assignment_pairs (list[AssignmentPair], optional): Expected assignment pairs. Used in debugging.

        Returns:
            StructurallyIdenticalResult
        """
        log10_max_num_assignment = np.log10(max_num_assignment)
        reference_reactant_nmat, reference_product_nmat = self.getMatricesForIdentity(identity)
        target_reactant_nmat, target_product_nmat = target.getMatricesForIdentity(identity)
        #####
        def makeAssignmentArr(cls:type)->Tuple[np.ndarray[int], bool]:  # type: ignore
           reference_constraint = cls(reference_reactant_nmat, reference_product_nmat, is_subset=is_subset)
           target_constraint = cls(target_reactant_nmat, target_product_nmat, is_subset=is_subset)
           compatibility_collection = reference_constraint.makeCompatibilityCollection(target_constraint) 
           compatibility_collection, is_truncated = compatibility_collection.prune(log10_max_num_assignment)
           return compatibility_collection.expand(), is_truncated
        #####
        # Construct the constraints for this comparison
        species_assignment_arr, is_species_truncated = makeAssignmentArr(SpeciesConstraint)
        reaction_assignment_arr, is_reaction_truncated = makeAssignmentArr(ReactionConstraint)
        is_truncated = is_species_truncated or is_reaction_truncated
        if len(species_assignment_arr) == 0 or len(reaction_assignment_arr) == 0:
            if IS_DEBUG:
                import pdb; pdb.set_trace()
            return StructurallyIdenticalResult(assignment_pairs=[], is_truncated=is_truncated)
        # Find the compatible assignments for species and reactions
        if identity == cn.ID_WEAK:
            satisfy_arr = self._compare(self.standard_nmat.values, target.standard_nmat.values,
                  species_assignment_arr, reaction_assignment_arr)
        else:
            reactant_satisfy_arr = self._compare(reference_reactant_nmat.values, target_reactant_nmat.values,
                    species_assignment_arr, reaction_assignment_arr)
            product_satisfy_arr = self._compare(reference_product_nmat.values, target_product_nmat.values,
                    species_assignment_arr, reaction_assignment_arr)
            satisfy_arr = np.logical_and(reactant_satisfy_arr, product_satisfy_arr)
        # Construct the indices for reactions and species
        assignment_idxs = np.array(range(len(satisfy_arr)))
        num_reaction_assignment = reaction_assignment_arr.shape[0]
        species_assignment_idxs = assignment_idxs[satisfy_arr]//num_reaction_assignment
        reaction_assignment_idxs = np.mod(assignment_idxs[satisfy_arr], num_reaction_assignment)
        assignment_pairs = []
        for species_idx, reaction_idx in zip(species_assignment_idxs, reaction_assignment_idxs):
            if species_assignment_arr.ndim == 2:
                species_assignment = species_assignment_arr[species_idx, :]
            else:
                species_assignment = species_assignment_arr
            if reaction_assignment_arr.ndim == 2:
                reaction_assignment = reaction_assignment_arr[reaction_idx, :]
            else:
                reaction_assignment = reaction_assignment_arr
            assignment_pair = AssignmentPair(species_assignment=species_assignment,
                                            reaction_assignment=reaction_assignment)
            assignment_pairs.append(assignment_pair)
        # Construct the result
        if IS_DEBUG:
            if len(assignment_pairs) == 0:
                import pdb; pdb.set_trace()
        return StructurallyIdenticalResult(assignment_pairs=assignment_pairs,
                is_truncated=is_truncated)