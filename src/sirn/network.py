'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from sirn import constants as cn  # type: ignore
from sirn import util  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.network_base import NetworkBase  # type: ignore

import copy
import collections
import itertools
import numpy as np
from typing import Optional, Tuple


MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis
#  assignments: np.ndarray[np.ndarray[int]]
#  is_truncated: bool (True if the number of assignments exceeds the maximum number of assignments)
AssignmentResult = collections.namedtuple('AssignmentResult', 'assignment_arr is_truncated compression_factor')


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
          orientation:str=cn.OR_SPECIES,
          identity:str=cn.ID_WEAK,
          is_subsets:bool=False,
          )->list[list[int]]:
        """
        Constructs a vector of lists of rows in the target that are compatible with each row in the reference (self).
        Handles the interaction between identity and participant.

        Args:
            target_network (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.

        Returns:
            list-list: Vector of lists of rows in the target that are compatible with each row in the reference.
        """
        compatible_sets:list = []
        def makeSets(participant:Optional[str]=None):
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
            big_reference_arr = util.repeatRow(reference_matrix.values, target_num_row)
            big_target_arr = util.repeatArray(target_matrix.values, reference_num_row)
            # Find the compatible sets
            if is_subsets:
                big_compatible_arr = np.less_equal(big_reference_arr, big_target_arr)
            else:
                big_compatible_arr = np.equal(big_reference_arr, big_target_arr)
            satisfy_arr = np.sum(big_compatible_arr, axis=1) == num_criteria
            # Construct the sets
            target_indices = np.array(range(target_num_row))
            for iset in range(reference_num_row):
                indices = target_num_row*iset + target_indices
                compatible_sets[iset] = target_indices[satisfy_arr[indices]].tolist()
            return compatible_sets
        #
        # Check compatibility
        if (not is_subsets) and not self.isStructurallyCompatible(target):
            return compatible_sets
        if is_subsets and not self.isSubsetCompatible(target):
            return compatible_sets
        # Construct the compatibility sets
        if orientation == cn.OR_REACTION:
            this_num_row = self.num_reaction
        else:
            this_num_row = self.num_species
        compatible_sets = [ [] for _ in range(this_num_row)]
        # Find compatible Rows
        if identity == cn.ID_WEAK:
            compatible_sets = makeSets(participant=None)
        else:
            reactant_compatible_sets = makeSets(participant=cn.PR_REACTANT)
            product_compatible_sets = makeSets(participant=cn.PR_PRODUCT)
            for idx in range(this_num_row):
                compatible_sets[idx] = list(set(reactant_compatible_sets[idx]) & set(product_compatible_sets[idx]))
        #
        return compatible_sets
    
    def _isCompatibleMatrices(self, reference_mat:Matrix, target_mat:Matrix, is_subsets:bool)->bool:
        """
        Determines if the matrices are compatible.

        Args:
            reference_mat (Matrix): Reference matrix.
            target_mat (Matrix): Target matrix.
            is_subsets (bool): If True, check for subsets of other.

        Returns:
            bool: True if compatible.
        """
        if is_subsets:
            return np.less_equal(reference_mat.values, target_mat.values).all()
        return np.equal(reference_mat.values, target_mat.values).all()

    def makeCompatibleAssignments(self,
                                  target:'Network',
                                  orientation:str=cn.OR_SPECIES,
                                  identity:str=cn.ID_WEAK,
                                  is_subsets:bool=False,
                                  max_num_assignment=cn.MAX_NUM_ASSIGNMENT)->AssignmentResult:
        """
        Constructs a list of compatible assignments. The strategy is to find initial segments
        that are pairwise compatible. Handles strong vs. weak identity by considering both reactant
        and product PairCriteriaCountMatrix for strong identity. At any step in the assignment process,
        the number of assignments is pruned if it exceeds the maximum number of assignments. The pruning
        is done by selecting a random subset of assignments.

        Args:
            target (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            is_subsets (bool): If True, check for subsets of other.
            max_num_assignment (int): Maximum number of assignments.

        Returns:
            AssignmentResult
        """
        def checkAssignments(assignment_arr:np.ndarray, participant:Optional[str]=None)->np.ndarray:
                """
                Checks if the assignments are compatible with the PairwiseCriteriaCountMatrices of
                the reference and target matrices.
                """
                # Get the pair array for the last two columns of the assignment array
                reference_pair_criteria_matrix = self.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
                target_pair_criteria_matrix = target.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
                num_assignment = assignment_arr.shape[0]
                pair_arr = assignment_arr[:, -2:]
                target_arr = target_pair_criteria_matrix.getTargetArrayFromPairArray(pair_arr[:, 0], pair_arr[:, 1])
                reference_arr = np.vstack(
                    [reference_pair_criteria_matrix.values[pos-1, pos, :].tolist()*num_assignment]).T
                num_column = target_arr.shape[1]
                reference_arr = np.reshape(reference_arr, (num_assignment, num_column))
                if is_subsets:
                    compatible_arr = np.less_equal(reference_arr, target_arr)
                else:
                    compatible_arr = np.equal(reference_arr, target_arr)
                # Find the rows that are compatible
                satisfy_arr = np.sum(compatible_arr, axis=1) == num_column
                assignment_arr = assignment_arr[satisfy_arr, :]
                return assignment_arr   
        #
        def prune(assignment_arr, expansion_factor:Optional[int]=None)->Tuple[np.ndarray, bool]:
            """Prunes the assignments if the number of assignments exceeds the maximum number of assignments.

            Args:
                assignment_arr (_type_): _description_

            Returns:
                Tuple[np.ndarray, bool]: _description_
            """
            is_truncated = False
            if expansion_factor is None:
                keep_count = max_num_assignment
            else:
                # Considers expansion of assignments by set cross product
                keep_count = int(assignment_arr.shape[0]*(1 - 1/expansion_factor))
                keep_count = int(max_num_assignment/expansion_factor)
            if assignment_arr.shape[0] > keep_count:
                select_idx = np.random.choice(assignment_arr.shape[0], keep_count, replace=False)
                assignment_arr = assignment_arr[select_idx, :]
                is_truncated = True
            return assignment_arr, is_truncated
        #
        compatible_sets = self.makeCompatibilitySetVector(target, orientation=orientation, identity=identity,
                                                          is_subsets=is_subsets)
        assignment_len = len(compatible_sets)  # Number of rows in the reference
        # Initialize the 2d array of assignments. Rows are assignment instance and columns are rows in the target.
        assignment_arr = np.array(np.array(compatible_sets[0]))
        assignment_arr = np.reshape(assignment_arr, (len(assignment_arr), 1))
           
        # Incrementally extend assignments by the cross product of the compatibility set for each position
        initial_sizes:list = []  # Number of assignments at each step
        final_sizes:list = [] # Number of assignments after cross product and pruning 
        is_truncated = False # True if the set of assignments was truncated because of excessive size
        for pos in range(1, assignment_len):
            # Check if no assignments
            if assignment_arr.shape[0] == 0:
                return AssignmentResult(assignment_arr=assignment_arr, is_truncated=is_truncated,
                                compression_factor=np.array(initial_sizes)/np.array(final_sizes))
            # Prune the number of assignments if it exceeds the maximum number of assignments
            prune_factor = len(compatible_sets[pos])
            assignment_arr, new_is_truncated = prune(assignment_arr, expansion_factor=prune_factor)
            is_truncated = is_truncated or new_is_truncated
            # Extend assignment_arr to the cross product of the compatibility set for this position.$a
            #   This is done by repeating by doing a block repeat of each row in the assignment array
            #   with a block size equal to the number of elements in the compatibility set for compatibility_sets[pos],
            #   and then repeating the compatibility set block for compatibility_sets[pos] for each
            #   row in the assignment array.
            ex_assignment_arr = np.repeat(assignment_arr, len(compatible_sets[pos]), axis=0)
            ex_compatibility_arr = np.vstack([compatible_sets[pos]*len(assignment_arr)]).T
            assignment_arr = np.hstack((ex_assignment_arr, ex_compatibility_arr))
            initial_sizes.append(assignment_arr.shape[0])
            # Find the rows that have a duplicate value. We do this by: (a) sorting the array, (b) finding the
            #   difference between the array and the array shifted by one, (c) finding the product of the differences.
            #   The product of the differences will be zero if there is a duplicate value.
            sorted_arr = np.sort(assignment_arr, axis=1)
            diff_arr = np.diff(sorted_arr, axis=1)
            prod_arr = np.prod(diff_arr, axis=1)
            keep_idx = np.where(prod_arr > 0)[0]
            assignment_arr = assignment_arr[keep_idx, :]
            # Check for pairwise compatibility between the last assignment and the new assignment
            #   This is done by finding the pairwise column for the previous and current assignment.
            #   Then we verify that the reference row is either equal (is_subset=False) or less than or equal (is_subset=True)
            #   the target row.
            if identity == cn.ID_WEAK:
                assignment_arr = checkAssignments(assignment_arr, None)
            else:
                # Strong identity
                # Must satisfy conditions for both reactant and product
                assignment_arr = checkAssignments(assignment_arr, participant=cn.PR_REACTANT)
                assignment_arr = checkAssignments(assignment_arr, participant=cn.PR_PRODUCT)
            final_sizes.append(assignment_arr.shape[0])
        #
        return AssignmentResult(assignment_arr=assignment_arr, is_truncated=is_truncated,
                                compression_factor=np.array(initial_sizes)/np.array(final_sizes))

    def _reduceCompatibleSets(self, compatible_sets:list, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->list:
            """
            Reduces the compatible sets so that the maximum number of assignments is not exceeded.

            Args:
                compatible_sets (list): List of compatible sets.

            Returns:
                list: List of reduced compatible sets.
            """
            new_sets = copy.deepcopy(compatible_sets)
            values = []
            [values.extend(s) for s in new_sets]  # type: ignore
            num_value = len(values)
            #
            for _ in range(num_value):
                sizes = [len(s) for s in new_sets]
                num_assignment = np.prod(sizes)
                if num_assignment < max_num_assignment:
                    break
                i_large = sizes.index(max(sizes))
                sel_set = new_sets[i_large]
                i_pos = np.random.randint(0, len(sel_set))
                reduced_sel_set = sel_set[:i_pos] + sel_set[i_pos+1:]
                new_sets[i_large] = reduced_sel_set
            return new_sets