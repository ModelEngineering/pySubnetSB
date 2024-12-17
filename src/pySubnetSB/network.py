'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB import util  # type: ignore
from pySubnetSB.matrix import Matrix  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint   # type: ignore
from pySubnetSB.network_base import NetworkBase, AssignmentPair  # type: ignore
from pySubnetSB.assignment_evaluator import AssignmentEvaluator  # type: ignore

import pynauty  # type: ignore
import numpy as np
from typing import Optional, List, Tuple, Union


NULL_ARRAY = np.array([])  # Null array
ATTRS = ["reactant_nmat", "product_nmat", "reaction_names", "species_names", "network_name"]
MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis


class StructuralAnalysisResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
                 assignment_pairs:list[AssignmentPair],
                 is_truncated:Optional[bool]=False,
                 num_species_candidate:int=-1,
                 num_reaction_candidate:int=-1,
                 network:Optional['Network']=None,
                 )->None:
        """
        Args:
            assignment_pairs (list[AssignmentPair]): List of assignment pairs.
            is_trucnated (bool): True if the number of assignments exceeds the maximum number of assignments.
            num_species_candidate (int): Number of species candidates assignments
            num_reaction_candidate (int): Number of reaction candidates assignments.
        """
        self.assignment_pairs = assignment_pairs
        self.is_truncated = is_truncated
        self.num_species_candidate = num_species_candidate
        self.num_reaction_candidate = num_reaction_candidate
        self.network = network

    @property
    def induced_network(self)->'Network':
        """Induced network from the first assignment pair."""
        return self.makeInducedNetwork()

    def makeInducedNetwork(self, assignment_pair_idx:int=0)->'Network':
        """
        Creates an induced network from the assignment pair.

        Args:
            assignment_pair_idx (int): index of the assignment pair

        Returns:
            str
        """
        if self.network is None:
            raise ValueError("Network is not defined.")
        if len(self.assignment_pairs) <= assignment_pair_idx:
            msg = f'Assignment pair index {assignment_pair_idx} is out of range.'
            msg += f' Max is {len(self.assignment_pairs)}'
            raise ValueError(msg)
        return self.network.makeInducedNetwork(self.assignment_pairs[assignment_pair_idx])

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
            reactant_arr (np.ndarray): Reactant matrix.
            product_arr (np.ndarray): Product matrix.
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

    def isStructurallyIdentical(self, target:'Network', is_subnet:bool=True, num_process:int=-1,
            max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
            max_batch_size:int=cn.MAX_BATCH_SIZE, identity:str=cn.ID_WEAK,
            is_report:bool=True, is_return_if_truncated:bool=True)->StructuralAnalysisResult:
        """
        Determines if the network is structurally identical to another network or subnet of another network.

        Args:
            target (Network): Network to search for structurally identity
            num_process (int, optional): Number of processes (default: -1 is all)
            is_subnets (bool, optional): Consider subsets
            max_num_assignment (int, optional): Maximum number of assignments to search (no limit if negative)
            max_batch_size (int, optional): Maximum batch size
            identity (str, optional): cn.ID_WEAK or cn.ID_STRONG
            is_report (bool, optional): Print report
            is_return_if_truncated (bool, optional): Return if truncation is required

        Returns:
            StructurallyIdenticalResult
        """
        if self.num_reaction == 0 or target.num_reaction == 0:
              return StructuralAnalysisResult(assignment_pairs=[], 
                  num_reaction_candidate=0,
                  num_species_candidate=0,
                  is_truncated=False)
        # Initialization
        if max_num_assignment < 0:
            log10_max_num_assignment = np.inf
        else:
            log10_max_num_assignment = np.log10(max_num_assignment)
        reference_reactant_nmat, reference_product_nmat = self.makeMatricesForIdentity(identity)
        target_reactant_nmat, target_product_nmat = target.makeMatricesForIdentity(identity)
        #####
        def makeAssignmentArr(cls:type)->Tuple[np.ndarray[int], bool, bool]:  # type: ignore
            reference_constraint = cls(reference_reactant_nmat, reference_product_nmat, is_subnet=is_subnet)
            target_constraint = cls(target_reactant_nmat, target_product_nmat, is_subnet=is_subnet)
            compatibility_collection = reference_constraint.makeCompatibilityCollection(
                  target_constraint).compatibility_collection
            compatibility_collection, prune_is_truncated = compatibility_collection.prune(log10_max_num_assignment)
            is_null = compatibility_collection.log10_num_assignment == -np.inf
            if is_null:
                return NULL_ARRAY, prune_is_truncated, is_null
            else:
                assignment_arr, expand_is_truncated = compatibility_collection.expand(max_num_assignment=max_num_assignment)
                if assignment_arr is NULL_ARRAY:
                    return NULL_ARRAY, expand_is_truncated, is_null
                if assignment_arr.ndim < 2:
                    return NULL_ARRAY, expand_is_truncated, is_null
                if assignment_arr.shape[1] == 0:
                    return NULL_ARRAY, expand_is_truncated, is_null
                is_truncated = prune_is_truncated or expand_is_truncated
                return assignment_arr, is_truncated, is_null
        #####
        # Calculate the compatibility vectors for species and reactions and then construct the assignment arrays
        species_assignment_arr, is_species_truncated, is_species_null = makeAssignmentArr(SpeciesConstraint)
        reaction_assignment_arr, is_reaction_truncated, is_reaction_null = makeAssignmentArr(ReactionConstraint)
        is_truncated = is_species_truncated or is_reaction_truncated
        # Check if further truncation is required
        num_species_assignment = species_assignment_arr.shape[0]
        num_reaction_assignment = reaction_assignment_arr.shape[0]
        if num_species_assignment*num_reaction_assignment > max_num_assignment:
            is_truncated = True
            if is_return_if_truncated:
                return StructuralAnalysisResult(assignment_pairs=[], 
                  num_reaction_candidate=num_reaction_assignment,
                  num_species_candidate=num_species_assignment,
                  is_truncated=is_truncated)
            else:
                # Truncate the assignment arrays
                species_frac = num_species_assignment/max_num_assignment
                reaction_frac = num_reaction_assignment/max_num_assignment
                species_assignment_arr = util.selectRandom(species_assignment_arr, int(species_frac*max_num_assignment))
                reaction_assignment_arr = util.selectRandom(reaction_assignment_arr, int(reaction_frac*max_num_assignment))
        # Handle null assignment
        is_null = is_species_null or is_reaction_null
        if len(species_assignment_arr) == 0 or len(reaction_assignment_arr) == 0 or is_null:
            return StructuralAnalysisResult(assignment_pairs=[], 
                  num_reaction_candidate=reaction_assignment_arr.shape[0],
                  num_species_candidate=species_assignment_arr.shape[0],
                  is_truncated=is_truncated)
        # Evaluate the assignments
        #   Evaluate on single byte entries
        evaluator = AssignmentEvaluator(reference_reactant_nmat.values.astype(np.int8),
              target_reactant_nmat.values.astype(np.int8), max_batch_size=max_batch_size)
        reactant_assignment_pairs = evaluator.parallelEvaluate(species_assignment_arr, reaction_assignment_arr,
                total_process=num_process, is_report=is_report)
        #   Check assignment pairs on single bytes
        evaluator = AssignmentEvaluator(reference_reactant_nmat.values,
              target_reactant_nmat.values, max_batch_size=max_batch_size)
        reactant_assignment_pairs = evaluator.evaluateAssignmentPairs(reactant_assignment_pairs)
        #   Evaluate on product matrices
        evaluator = AssignmentEvaluator(reference_product_nmat.values, target_product_nmat.values,
            max_batch_size=max_batch_size)
        assignment_pairs = evaluator.evaluateAssignmentPairs(reactant_assignment_pairs)
        # Return result
        return StructuralAnalysisResult(assignment_pairs=assignment_pairs,
              num_reaction_candidate=reaction_assignment_arr.shape[0],
              num_species_candidate=species_assignment_arr.shape[0],
              is_truncated=is_truncated,
              network=target)