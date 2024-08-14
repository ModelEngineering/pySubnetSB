'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from sirn import constants as cn  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.network_base import NetworkBase  # type: ignore

import copy
import itertools
import numpy as np
from typing import Optional


MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis


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
          participant:Optional[str]=None,
          is_subsets:bool=False,
          )->list[list[int]]:
        """
        Constructs a vector of lists of rows in the target that are compatible with each row in the reference (self).

        Args:
            target_network (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.

        Returns:
            list-list: Vector of lists of rows in the target that are compatible with each row in the reference.
        """
        def makeBigArray(matrix:Matrix, other_num_row:int, is_block_repeats=False)->list:
            """
            Constructs a large array that allows for simultaneous comparison of all rows.
              is_rotate: True: rotate the rows in the array
                         False: block repetitions

            Args:
                matrix (Matrix): Matrix to expand.
                other_num_row (int): Number of rows in other matrix.
                is_block_repeats (bool): If True, block repeats.

            Returns:
                list-list
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
            return arr.tolist()
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
                big_compatible_arr = np.less_equal(big_reference_arr, big_target_arr)
            else:
                big_compatible_arr = np.equal(big_reference_arr, big_target_arr)
            satisfy_arr = np.sum(big_compatible_arr, axis=1) == num_criteria
            # Construct the sets
            target_indices = np.array(range(target_num_row))
            for iset in range(reference_num_row):
                indices = target_num_row*iset + target_indices
                compatible_sets[iset] = target_indices[satisfy_arr[indices]].tolist()
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
    
    def oldermakeCompatibleAssignments(self,
                                  target:'Network',
                                  orientation:str=cn.OR_SPECIES,
                                  identity:str=cn.ID_WEAK,
                                  participant:Optional[str]=None,
                                  is_subsets:bool=False,
                                  max_num_assignment=cn.MAX_NUM_ASSIGNMENT,
                                  max_prefix_len:int=MAX_PREFIX_LEN)->list[np.ndarray[int]]:
        """
        Constructs a list of compatible assignments. The strategy is to find prefixes that are pairwise compatible.

        Args:
            target (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.
            max_num_assignment (int): Maximum number of assignments.
            max_prefix_len (int): Maximum length of a prefix in the assignment to do a pairwise analysis.

        Returns:
            list: List of compatible assignments.
        """
        compatible_sets = self.makeCompatibilitySetVector(target, orientation=orientation, identity=identity,
                                                          participant=participant, is_subsets=is_subsets)
        assignment_len = len(compatible_sets)
        reference_pair_criteria_matrix = self.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
        target_pair_criteria_matrix = target.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
        # Build a list of prefixes that are used for the first MAX_PREFIX_LEN of an assignment
        if assignment_len > max_prefix_len:
            # Find compatiable prefixes based on a pairwise analysis
            good_prefixes:list = compatible_sets[0]
            for prefix_idx in range(1, max_prefix_len):
                reference_mat = reference_pair_criteria_matrix.getReferenceArray(assignment_len=prefix_idx+1)
                for prefix in good_prefixes:
                    for n in compatible_sets[prefix_idx]:
                        assignment = prefix + [n]
                        target_mat = target_pair_criteria_matrix.getTargetArray(assignment_len=prefix_idx+1,
                                                                                assignment=prefix) 
                        if self._isCompatibleMatrices(reference_mat, target_mat, is_subsets):
                            good_prefixes.append(assignment)
        # Select the rest of the assignment
        tail_sets = compatible_sets[max_prefix_len:]
        new_max_num_assignment = max(max_num_assignment // len(good_prefixes), 10)
        reduced_tail_sets = self._reduceCompatibleSets(tail_sets, new_max_num_assignment)
        tail_assignments = list(itertools.product(*reduced_tail_sets))
        assignments = []
        for prefix in good_prefixes:
            for tail_assignment in tail_assignments:
                assignment = prefix + tail_assignment
                if len(assignment) == assignment_len:
                    assignments.append(prefix + tail_assignment)
        #
        return assignments
    

    def makeCompatibleAssignments(self,
                                  target:'Network',
                                  orientation:str=cn.OR_SPECIES,
                                  identity:str=cn.ID_WEAK,
                                  participant:Optional[str]=None,
                                  is_subsets:bool=False,
                                  max_num_assignment=cn.MAX_NUM_ASSIGNMENT,
                                  max_prefix_len:int=MAX_PREFIX_LEN)->np.ndarray[np.ndarray[int]]:
        """
        Constructs a list of compatible assignments. The strategy is to find prefixes that are pairwise compatible.

        Args:
            target (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.
            max_num_assignment (int): Maximum number of assignments.
            max_prefix_len (int): Maximum length of a prefix in the assignment to do a pairwise analysis.

        Returns:
            list: List of compatible assignments.
        """
        compatible_sets = self.makeCompatibilitySetVector(target, orientation=orientation, identity=identity,
                                                          participant=participant, is_subsets=is_subsets)
        assignment_len = len(compatible_sets)  # Number of rows in the reference
        reference_pair_criteria_matrix = self.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
        target_pair_criteria_matrix = target.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
        reference_sequential_matrix = reference_pair_criteria_matrix.getReferenceArray(assignment_len=assignment_len)
        # Initialize the matrix of assignments. Rows are assignment instance and columns are rows in the target.
        assignment_arr = np.array(np.array([compatible_sets[0]]))
        assignment_arr = np.reshape(assignment_arr, (len(assignment_arr), 1))
        # Incrementally extend the assignment matrix
        for pos in range(1, assignment_len):
            pass
            # Add a column that is the cross product of the compatibility set for this position.$a
            #   This is done by repeating by doing a block repeat of each row in the assignment array
            #   with a block size equal to the number of elements in the compatibility set for compatibility_sets[pos],
            #   and then repeating the compatibility set block for compatibility_sets[pos] for each row in the assignment array.

            # Remove assignments (rows) that have a repeated index in the target (using np.unique)

            # Check for pairwise compatibility between the last assignment and the new assignment
        #
        return assignment_arr


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