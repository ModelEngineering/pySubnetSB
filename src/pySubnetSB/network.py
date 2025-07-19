'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB import util  # type: ignore
from pySubnetSB.matrix import Matrix  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint   # type: ignore
from pySubnetSB.network_base import NetworkBase, AssignmentPair  # type: ignore
from pySubnetSB.assignment_evaluator import AssignmentEvaluator  # type: ignore
from pySubnetSB.performance_monitor import PerformanceMonitor  # type: ignore

import numpy as np # type: ignore
from typing import Optional, List, Tuple, Union


NULL_ARRAY = np.array([])  # Null array
ATTRS = ["reactant_nmat", "product_nmat", "reaction_names", "species_names", "network_name"]
MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis
IS_ENABLE_PERFORANCE_REPORTING = False


class StructuralAnalysisResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
            assignment_pairs:List[AssignmentPair],
            is_truncated:Optional[bool]=False,
            num_species_candidate:int=-1,
            num_reaction_candidate:int=-1,
            network:Optional['Network']=None,
            )->None:
        """
        Args:
            assignment_pairs (List[AssignmentPair]): List of assignment pairs.
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
    def mapping_pairs(self)->List[AssignmentPair]:
        """Same as assignment pairs."""
        return self.assignment_pairs

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
                        species_names=self.species_names)
    
    def isIsomorphic(self, target:'Network')->bool:
        """Using pynauty to detect isomorphism of reaction networks.

        Args:
            target (Network)

        Returns:
            bool
        """
        import pynauty  # type: ignore
        self_graph = self.makePynautyNetwork()
        target_graph = target.makePynautyNetwork()
        return pynauty.isomorphic(self_graph, target_graph)

    def isStructurallyIdentical(self, target:'Network', is_subnet:bool=True, num_process:int=-1,
            max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
            max_batch_size:int=cn.MAX_BATCH_SIZE, identity:str=cn.ID_WEAK,
            is_all_valid_assignment:bool=True,
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
            is_all_valid_assignment (bool, optional): Return all valid assignments
            is_return_if_truncated (bool, optional): Return if truncation is required

        Returns:
            StructurallyIdenticalResult
        """
        MIN_ASSIGNMENT_FOR_PARALLELISM = 1e7
        monitor = PerformanceMonitor("isStructurallyIdentical: Start", is_enabled=IS_ENABLE_PERFORANCE_REPORTING)
        monitor.add("isStructurallyIdentical/Initialization: Start")
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
        monitor.add("isStructurallyIdentical/Initialization: End")
        #####
        def makeAssignmentArr(cls:type)->Tuple[np.ndarray[int], bool, bool]:  # type: ignore
            monitor.add("makeAssignmentArr/makeAssignmentArr: Start")
            reference_constraint = cls(reference_reactant_nmat, reference_product_nmat, is_subnet=is_subnet)
            target_constraint = cls(target_reactant_nmat, target_product_nmat, is_subnet=is_subnet)
            monitor.add("makeAssignmentArr/makeAssignmentArr/Make compatibility collection: Start")
            compatibility_collection = reference_constraint.makeCompatibilityCollection(
                  target_constraint).compatibility_collection
            monitor.add("makeAssignmentArr/makeAssignmentArr/Make compatibility collection: Created")
            compatibility_collection, prune_is_truncated = compatibility_collection.prune(log10_max_num_assignment)
            monitor.add("makeAssignmentArr/makeAssignmentArr/Make compatibility collection: End")
            is_null = compatibility_collection.log10_num_assignment == -np.inf
            if is_null:
                monitor.add("makeAssignmentArr/makeAssignmentArr/Null return")
                return NULL_ARRAY, prune_is_truncated, is_null
            log10_num_assignment = compatibility_collection.log10_num_assignment
            if log10_num_assignment > 0.5*np.log10(max_num_assignment):
                msg = f"makeAssignmentArr/makeAssignmentArr/Assignment array too large[{log10_num_assignment}]"
                monitor.add(msg)
                return NULL_ARRAY, True, is_null
            else:
                msg = f"makeAssignmentArr/makeAssignmentArr/Expand[{log10_num_assignment}]: Start"
                monitor.add(msg)
                # FIXME: The following takes a long time
                assignment_arr, expand_is_truncated = compatibility_collection.expand(max_num_assignment=max_num_assignment)
                if assignment_arr is NULL_ARRAY:
                    return NULL_ARRAY, expand_is_truncated, is_null
                if assignment_arr.ndim < 2:
                    return NULL_ARRAY, expand_is_truncated, is_null
                if assignment_arr.shape[1] == 0:
                    return NULL_ARRAY, expand_is_truncated, is_null
                is_truncated = prune_is_truncated or expand_is_truncated
                monitor.add("makeAssignmentArr/makeAssignmentArr/Expand: End")
                return assignment_arr, is_truncated, is_null
        #####
        # Calculate the compatibility vectors for species and reactions and then construct the assignment arrays
        monitor.add("makeAssignmentArr/Make compatibility vectors: Start")
        species_assignment_arr, is_species_truncated, is_species_null = makeAssignmentArr(SpeciesConstraint)
        reaction_assignment_arr, is_reaction_truncated, is_reaction_null = makeAssignmentArr(ReactionConstraint)
        monitor.add("makeAssignmentArr/Make compatibility vectors: End")
        is_truncated = is_species_truncated or is_reaction_truncated
        # Check if further truncation is required
        size = species_assignment_arr.shape[0], reaction_assignment_arr.shape[0]
        msg = f"makeAssignmentArr/Check further truncation {size}: Start"
        monitor.add(msg)
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
        monitor.add("makeAssignmentArr/Check further truncation: End")
        # Handle null assignment
        monitor.add("makeAssignmentArr/Null assignment: Start")
        is_null = is_species_null or is_reaction_null
        if len(species_assignment_arr) == 0 or len(reaction_assignment_arr) == 0 or is_null:
            return StructuralAnalysisResult(assignment_pairs=[], 
                  num_reaction_candidate=reaction_assignment_arr.shape[0],
                  num_species_candidate=species_assignment_arr.shape[0],
                  is_truncated=is_truncated)
        monitor.add("makeAssignmentArr/Null assignment: End")
        # Evaluate the assignments
        size = species_assignment_arr.shape[0], reaction_assignment_arr.shape[0]
        msg = f"makeAssignmentArr/Evaluate reactant assignments for {size}: Start"
        monitor.add(msg)
        #   Evaluate the assignment pairs for the reactant stoichiometry matrix
        evaluator = AssignmentEvaluator(reference_reactant_nmat.values,
              target_reactant_nmat.values, max_batch_size=max_batch_size)
        total_num_assignment = species_assignment_arr.shape[0]*reaction_assignment_arr.shape[0]
        actual_num_process = num_process if total_num_assignment > MIN_ASSIGNMENT_FOR_PARALLELISM else 1
        result = evaluator.parallelEvaluate(species_assignment_arr, reaction_assignment_arr,
                total_process=actual_num_process, is_report=is_report, max_num_assignment=max_num_assignment)
        is_truncated = is_truncated or result.is_truncated
        reactant_assignment_pairs = result.assignment_pairs
        monitor.add("makeAssignmentArr/Evaluate reactant assignments: End")
        #   Evaluate on product matrices for the assignment pairs found for the reactant matrices
        monitor.add("makeAssignmentArr/Evaluate product assignments: Start")
        evaluator = AssignmentEvaluator(reference_product_nmat.values, target_product_nmat.values,
            max_batch_size=max_batch_size)
        bounded_max_num_assignment = max_num_assignment if is_all_valid_assignment else 1
        assignment_pairs = evaluator.evaluateAssignmentPairs(reactant_assignment_pairs,
              max_num_assignment=bounded_max_num_assignment)
        monitor.add("makeAssignmentArr/Evaluate product assignments: End")
        # Return result
        return StructuralAnalysisResult(assignment_pairs=assignment_pairs,
              num_reaction_candidate=reaction_assignment_arr.shape[0],
              num_species_candidate=species_assignment_arr.shape[0],
              is_truncated=is_truncated,
              network=target)