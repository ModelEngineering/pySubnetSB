'''Central class in the DISRN algorithm. Does analysis of network structures.'''

from sirn import constants as cn  # type: ignore
from sirn import util  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.network_base import NetworkBase, AssignmentPair, CRITERIA_VECTOR  # type: ignore

import collections
import pynauty  # type: ignore
import numpy as np
from typing import Optional, List, Tuple, Union


IS_DEBUG = False

NULL_ARRAY = np.array([])  # Null array
ATTRS = ["reactant_mat", "product_mat", "reaction_names", "species_names", "network_name"]


MAX_PREFIX_LEN = 3   # Maximum length of a prefix in the assignment to do a pairwise analysis
#  assignments: np.ndarray[np.ndarray[int]]
#  is_truncated: bool (True if the number of assignments exceeds the maximum number of assignments)
AssignmentResult = collections.namedtuple('AssignmentResult', 'assignment_arr is_truncated compression_factor')
# Pair of assignments of species and reactions in target network to the reference network.


class StructurallyIdenticalResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
                 assignment_pairs:list[AssignmentPair],
                 is_truncated:Optional[bool]=False,
                 species_compression_factor:Optional[float]=None,
                 reaction_compression_factor:Optional[float]=None,
                 )->None:
        """
        Args:
            assignment_pairs (list[AssignmentPair]): List of assignment pairs.
            is_trucnated (bool): True if the number of assignments exceeds the maximum number of assignments.
            species_compression_factor (float): Number of species assignments considered at each step divided by the 
                number of assignments kept
            reaction_compression_factor (float): Number of species assignments considered at each step divided by the 
                number of assignments kept
        """
        self.assignment_pairs = assignment_pairs
        self.is_truncated = is_truncated
        self.species_compression_factor = species_compression_factor
        self.reaction_compression_factor = reaction_compression_factor

    def __bool__(self)->bool:
        return len(self.assignment_pairs) > 0
    
    def __repr__(self)->str:
        repr = f"StructurallyIdenticalResult(assignment_pairs={self.assignment_pairs};"
        repr += f" is_truncated={self.is_truncated};"
        repr += f" species_compression_factor={self.species_compression_factor};"
        repr += f" reaction_compression_factor={self.reaction_compression_factor})."
        return repr


class Network(NetworkBase):

    def __init__(self, reactant_arr:Union[np.ndarray, Matrix], 
                 product_arr:Union[np.ndarray, Matrix],
                 reaction_names:Optional[np.ndarray[str]]=None, # type: ignore
                 species_names:Optional[np.ndarray[str]]=None,  # type: ignore
                 network_name:Optional[str]=None,               # type: ignore
                 criteria_vector:Optional[CriteriaVector]=CRITERIA_VECTOR)->None:
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
                            reaction_names=reaction_names, species_names=species_names,
                            criteria_vector=criteria_vector) 
        
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
        return Network(self.reactant_mat.values.copy(), self.product_mat.values.copy(),
                        network_name=self.network_name,
                        reaction_names=self.reaction_names,
                        species_names=self.species_names,
                        criteria_vector=self.criteria_vector)
    
    def _validateAssignments(self, assignment_arr:np.ndarray[int], pos:int,  # type: ignore
          expected_assignment_arr:Optional[np.ndarray]=None):
        # Checks if the assignment is consistent with expectations
        if not IS_DEBUG:
            return
        if expected_assignment_arr is not None:
            truncated_assignment_arr = np.array([a[:pos+1] for a in assignment_arr])
            if not self._isMemberOfArray(expected_assignment_arr[:pos+1], truncated_assignment_arr):
                df = util.arrayToSortedDataFrame(truncated_assignment_arr)
                import pdb; pdb.set_trace()
    
    def makeCompatibilitySetVector(self,
          target:'Network',
          orientation:str=cn.OR_SPECIES,
          identity:str=cn.ID_WEAK,
          is_subsets:bool=False,
          expected_assignment_arr:Optional[np.ndarray]=None)->List[List[int]]:
        """
        Constructs a vector of lists of rows in the target that are compatible with each row in the reference (self).
        Handles the interaction between identity and participant.

        Args:
            target_network (Network): Target network.
            orientation (str): Orientation of the network. cn.OR_REACTIONS or cn.OR_SPECIES.
            identity (str): Identity of the network. cn.ID_WEAK or cn.ID_STRONG.
            participant (str): Participant in the network for ID_STRONG. cn.PR_REACTANT or cn.PR_PRODUCT.
            is_subsets (bool): If True, check for subsets of other.
            expected_assignment_arr (np.ndarray): Expected assignment. Used in debugging.

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
                #big_compatible_arr = np.isclose(big_reference_arr, big_target_arr)
                big_compatible_arr = big_reference_arr == big_target_arr
            satisfy_arr = np.sum(big_compatible_arr, axis=1) == num_criteria
            # Construct the sets
            target_indices = np.array(range(target_num_row))
            for iset in range(reference_num_row):
                indices = target_num_row*iset + target_indices
                compatible_sets[iset] = target_indices[satisfy_arr[indices]].tolist()
            if IS_DEBUG and expected_assignment_arr is not None:
                trues = [expected_assignment_arr[iset] in compatible_sets[iset] for iset in range(reference_num_row)]
                if not all(trues):
                    import pdb; pdb.set_trace()
            return compatible_sets
        #
        # Check compatibility
        if (not is_subsets) and not self.isStructurallyCompatible(target, identity=identity):
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

    def makeCompatibleAssignments(self,
                                  target:'Network',
                                  orientation:str=cn.OR_SPECIES,
                                  identity:str=cn.ID_WEAK,
                                  is_subsets:bool=False,
                                  max_num_assignment=cn.MAX_NUM_ASSIGNMENT,
                                  expected_assignment_arr:Optional[np.ndarray]=None)->AssignmentResult:
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
            expected_assignment (np.ndarray): Expected assignment. Used in debugging.

        Returns:
            AssignmentResult
        """
        #####
        def checkAssignments(assignment_arr:np.ndarray, participant:Optional[str]=None)->np.ndarray:
                """
                Checks if the assignments are compatible with the PairwiseCriteriaCountMatrices of
                the reference and target matrices.
                """
                if assignment_arr.shape[0] == 0:
                    return assignment_arr
                # Get the pair array for the last two columns of the assignment array
                reference_pair_criteria_matrix = self.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
                target_pair_criteria_matrix = target.getNetworkMatrix(matrix_type=cn.MT_PAIR_CRITERIA,
                                                                orientation=orientation,
                                                                identity=identity,
                                                                participant=participant)
                num_assignment, assignment_len = assignment_arr.shape
                target_arr = target_pair_criteria_matrix.getTargetArray(assignment_arr)
                base_reference_arr = reference_pair_criteria_matrix.getReferenceArray(assignment_len)
                #reference_arr = np.vstack([base_reference_arr]*num_assignment)
                reference_arr = np.concatenate([base_reference_arr]*num_assignment, axis=0)
                num_column = reference_pair_criteria_matrix.num_column
                if is_subsets:
                    compatible_arr = np.less_equal(reference_arr, target_arr)
                else:
                    #compatible_arr = np.isclose(reference_arr, target_arr)
                    compatible_arr = reference_arr == target_arr
                # Find the rows that are compatible
                satisfy_arr = np.sum(compatible_arr, axis=1) == num_column
                # Reshape so can count satisfying each position in the assignment
                satisfy_arr = np.reshape(satisfy_arr, (num_assignment, assignment_len-1))
                satisfy_arr = np.sum(satisfy_arr, axis=1) == assignment_len - 1
                new_assignment_arr = assignment_arr[satisfy_arr, :]
                return new_assignment_arr   
        #####
        is_truncated = False # True if the set of assignments was truncated because of excessive size
        compatible_sets = self.makeCompatibilitySetVector(target, orientation=orientation, identity=identity,
              is_subsets=is_subsets, expected_assignment_arr=expected_assignment_arr)
        assignment_len = len(compatible_sets)  # Number of rows in the reference
        # Initialize the 2d array of assignments. Rows are assignment instance and columns are rows in the target.
        if len(compatible_sets) == 0:
            if IS_DEBUG:
                import pdb; pdb.set_trace()
            return AssignmentResult(assignment_arr=NULL_ARRAY, is_truncated=is_truncated, compression_factor=None)
        if compatible_sets[0] == 0:
            if IS_DEBUG:
                import pdb; pdb.set_trace()
            return AssignmentResult(assignment_arr=NULL_ARRAY, is_truncated=is_truncated, compression_factor=None)
        assignment_arr = np.array(np.array(compatible_sets[0]))
        assignment_arr = np.reshape(assignment_arr, (len(assignment_arr), 1))
        self._validateAssignments(assignment_arr, 0, expected_assignment_arr)
        # Incrementally extend assignments by the cross product of the compatibility set for each position
        initial_sizes:list = []  # Number of assignments at each step
        final_sizes:list = [] # Number of assignments after cross product and pruning 
        for pos in range(1, assignment_len):
            # Check if no assignments
            if assignment_arr.shape[0] == 0:
                return AssignmentResult(assignment_arr=NULL_ARRAY, is_truncated=is_truncated, compression_factor=None)
            # Prune the number of assignments if it exceeds the maximum number of assignments
            keep_count = max_num_assignment // len(compatible_sets[pos])
            self._validateAssignments(assignment_arr, pos-1, expected_assignment_arr)
            try:
                new_assignment_arr, new_is_truncated = util.pruneArray(assignment_arr, keep_count)
            except:
                import pdb; pdb.set_trace()
            is_truncated = is_truncated or new_is_truncated
            self._validateAssignments(assignment_arr, pos-1, expected_assignment_arr)
            assignment_arr = new_assignment_arr
            if assignment_arr.shape[0] == 0:
                return AssignmentResult(assignment_arr=NULL_ARRAY, is_truncated=is_truncated, compression_factor=None)
            # Extend assignment_arr to the cross product of the compatibility set for this position.$a
            #   This is done by repeating by doing a block repeat of each row in the assignment array
            #   with a block size equal to the number of elements in the compatibility set for compatibility_sets[pos],
            #   and then repeating the compatibility set block for compatibility_sets[pos] for each
            #   row in the assignment array.
            ex_assignment_arr = np.repeat(assignment_arr, len(compatible_sets[pos]), axis=0)
            #ex_compatibility_arr = np.vstack([compatible_sets[pos]*len(assignment_arr)]).T
            ex_compatibility_arr = np.concatenate([compatible_sets[pos]*len(assignment_arr)], axis=0)
            ex_compatibility_arr = np.reshape(ex_compatibility_arr, (len(assignment_arr)*len(compatible_sets[pos]), 1))
            #assignment_arr = np.hstack((ex_assignment_arr, ex_compatibility_arr))
            assignment_arr = np.concatenate([ex_assignment_arr, ex_compatibility_arr], axis=1)
            self._validateAssignments(assignment_arr, pos, expected_assignment_arr)
            initial_sizes.append(assignment_arr.shape[0])
            # Find the rows that have a duplicate value. We do this by: (a) sorting the array, (b) finding the
            #   difference between the array and the array shifted by one, (c) finding the product of the differences.
            #   The product of the differences will be zero if there is a duplicate value.
            sorted_arr = np.sort(assignment_arr, axis=1)
            diff_arr = np.diff(sorted_arr, axis=1)
            prod_arr = np.prod(diff_arr, axis=1)
            keep_idx = np.where(prod_arr > 0)[0]
            assignment_arr = assignment_arr[keep_idx, :]
            self._validateAssignments(assignment_arr, pos, expected_assignment_arr)
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
            self._validateAssignments(assignment_arr, pos, expected_assignment_arr)
            final_sizes.append(assignment_arr.shape[0])
        #
        if all([s > 0 for s in final_sizes]):
            compression_factor = np.array(initial_sizes)/np.array(final_sizes)
        else:
            compression_factor = None
        return AssignmentResult(assignment_arr=assignment_arr, is_truncated=is_truncated,
                                compression_factor=compression_factor)
    
    @staticmethod
    def _isMemberOfArray(small_arr:np.ndarray, big_arr:np.ndarray)->bool:
        """
        Determines if a one dimensional array (small_arr) is a member of a two dimensional array (big_arr).

        Args:
            small_arr (np.ndarray): _description_
            big_arr (np.ndarray): _description_

        Returns:
            bool
        """
        return bool(np.any(np.all(big_arr == small_arr, axis=1)))
    
    def _makeCompatibilityVectorPermutedNetwork(self, target:'Network', identity:str, is_subsets:bool
              )->Tuple['Network', AssignmentPair]:
        """
        Creates a permuted network that is compatible with the target network such that
        species are ordered by the size of their compatibility sets with the target network
        and similarly reactions are ordered by the size of their compatibility sets with the target network.

        Args:
            target (Network): _description_

        Returns:
            Network: _description_
        """
        species_compatible_sets = self.makeCompatibilitySetVector(target, orientation=cn.OR_SPECIES,
              identity=identity, is_subsets=is_subsets)
        reaction_compatible_sets = self.makeCompatibilitySetVector(target, orientation=cn.OR_REACTION,
              identity=identity, is_subsets=is_subsets)
        species_perm = np.argsort([len(s) for s in species_compatible_sets])
        reaction_perm = np.argsort([len(r) for r in reaction_compatible_sets])
        permutation_pair = AssignmentPair(species_assignment=species_perm, reaction_assignment=reaction_perm)
        network, assignment_pair = self.permute(assignment_pair=permutation_pair)
        return network, assignment_pair
    
    def isStructurallyIdentical(self, target:'Network', is_subsets:bool=False,
            max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT, identity:str=cn.ID_WEAK,
            expected_assignment_pair:Optional[AssignmentPair]=None)->StructurallyIdenticalResult:
        """
        Determines if the network is structurally identical to another network.

        Args:
            target (Network): Network to search for structurally identity
            is_subsets (bool, optional): Consider subsets
            max_num_assignment (int, optional): Maximum number of assignments to produce.
            identity (str, optional): cn.ID_WEAK or cn.ID_STRONG
            expected_assignment_pairs (list[AssignmentPair], optional): Expected assignment pairs. Used in debugging.

        Returns:
            StructurallyIdenticalResult: _description_
        """
        # Construct a new reference network such that the compatibility sets with the reference
        # are ordered by increasing size. This is a heuristic to reduce the number of assignments.
        permuted_reference, inverse_permutation_pair = self._makeCompatibilityVectorPermutedNetwork(target,
              identity, is_subsets)
        reaction_perm = np.argsort(inverse_permutation_pair.reaction_assignment)
        species_perm = np.argsort(inverse_permutation_pair.species_assignment)
        # Initializations
        if expected_assignment_pair is not None:
            expected_reaction_assignment_arr = expected_assignment_pair.reaction_assignment[reaction_perm]
            expected_species_assignment_arr = expected_assignment_pair.species_assignment[species_perm]
        else:
            expected_reaction_assignment_arr = None
            expected_species_assignment_arr = None
        # Get the compatible assignments for species and reactions
        species_assignment_result = permuted_reference.makeCompatibleAssignments(target, cn.OR_SPECIES, identity=identity,
              is_subsets=is_subsets, max_num_assignment=max_num_assignment,
              expected_assignment_arr=expected_species_assignment_arr)
        species_assignment_arr = species_assignment_result.assignment_arr
        reaction_assignment_result = permuted_reference.makeCompatibleAssignments(target,
              cn.OR_REACTION, identity=identity,
              is_subsets=is_subsets, max_num_assignment=max_num_assignment,
              expected_assignment_arr=expected_reaction_assignment_arr)
        reaction_assignment_arr = reaction_assignment_result.assignment_arr
        is_truncated = species_assignment_result.is_truncated or reaction_assignment_result.is_truncated
        if len(species_assignment_arr) == 0 or len(reaction_assignment_arr) == 0:
            if IS_DEBUG:
                import pdb; pdb.set_trace()
            return StructurallyIdenticalResult(assignment_pairs=[], is_truncated=is_truncated)
        #####
        def compare(participant:Optional[str]=None)->Tuple[np.ndarray[bool], bool]:  # type: ignore
            """
            Compares the reference matrix to the target matrix for the participant and identity.

            Args:
                participant: cn.PR_REACTANT or cn.PR_PRODUCT

            Returns:
                np.ndarray[bool]: flattened boolean array of the outcome of comparisons for pairs of assignments
                                  (i.e., assignments of target species to reference species
                                  and target reactions to reference reactions).
                                  The array is organized breadth-first (column first).
                bool: True if the number of assignments exceeds the maximum number of assignments.
            """
            is_truncated = False
            reference_matrix = permuted_reference.getNetworkMatrix(matrix_type=cn.MT_STOICHIOMETRY,
                  orientation=cn.OR_SPECIES, participant=participant, identity=identity)
            target_matrix = target.getNetworkMatrix(matrix_type=cn.MT_STOICHIOMETRY, orientation=cn.OR_SPECIES,
                  participant=participant, identity=identity)
            # Size the comparison arrays used to make comparisons
            species_assignment_arr =  species_assignment_result.assignment_arr
            reaction_assignment_arr =  reaction_assignment_result.assignment_arr
            num_species_assignment =  len(species_assignment_arr)
            num_reaction_assignment =  len(reaction_assignment_arr)
            num_assignment =  num_species_assignment*num_reaction_assignment
            # Adjust assignments if size is excessive
            if num_assignment > max_num_assignment:
                is_truncated = True
                frac = max_num_assignment/num_assignment  # Fraction of assignments to keep
                num_species_assignment = int(num_species_assignment*frac)
                num_reaction_assignment = int(num_reaction_assignment*frac)
                species_assignment_arr, species_truncated = util.pruneArray(
                      species_assignment_arr, num_species_assignment)
                reaction_assignment_arr, reaction_truncated = util.pruneArray(
                      reaction_assignment_arr, num_reaction_assignment)
                is_truncated = is_truncated or species_truncated or reaction_truncated
                num_assignment =  num_species_assignment*num_reaction_assignment
            # Set up the comparison arrays. These are referred to as 'big' arrays.
            #big_reference_arr = np.vstack([reference_matrix.values]*num_assignment)
            big_reference_arr = np.concatenate([reference_matrix.values]*num_assignment, axis=0)
            species_idxs = species_assignment_arr.flatten()
            reaction_idxs = reaction_assignment_arr.flatten()
            #   The following constructs the re-arranged target matrix.
            #   This is organized as blocks of row rearrangements (species) and
            #   each row rearrangement is repeated for several columns (reactions) rearrangements.
            #   Indices are pair to select each row and column in the target matrix
            # 
            #   Create the sequence of reaction indices that is paired with the sequence of species indices
            #   This repeat the reaction indices for each species (row), and does this for each
            #   reaction assignment.
            structured_reaction_idxs = np.reshape(reaction_idxs, (num_reaction_assignment,
                  permuted_reference.num_reaction))
            big_structured_reaction1_idxs = np.concatenate([structured_reaction_idxs]*permuted_reference.num_species,
                  axis=1)
            big_structured_reaction2_idxs = np.concatenate([big_structured_reaction1_idxs]*num_species_assignment,
                  axis=0)
            big_reaction_idxs = big_structured_reaction2_idxs.flatten()
            #  Create the sequence of species indices that is paired with the sequence of reaction indices
            big_species1_idxs = np.repeat(species_idxs, permuted_reference.num_reaction)
            #  Reshape to the shape of the stoichiometry matrix
            big_species2_idxs = np.reshape(big_species1_idxs,
                  (num_species_assignment, permuted_reference.num_species*permuted_reference.num_reaction))
            #  Replicate for each reaction assignment
            big_species3_idxs = np.concatenate([big_species2_idxs]*num_reaction_assignment,
                  axis=1)
            #  Convert to a single array
            big_species_idxs = big_species3_idxs.flatten()
            #   The following constructs the re-arranged target matrix.
            big_target_arr = np.reshape(target_matrix.values[big_species_idxs, big_reaction_idxs],
                  (num_assignment*permuted_reference.num_species, permuted_reference.num_reaction))
            # Compare each row
            big_compatible_arr = np.isclose(big_reference_arr, big_target_arr)
            big_row_sum = np.sum(big_compatible_arr, axis=1)
            big_row_satisfy = big_row_sum == permuted_reference.num_reaction
            # Rows are results of the comparison of the reference and target; columns are assignments
            assignment_pair_satisfy_arr = np.reshape(big_row_satisfy, (num_assignment, permuted_reference.num_species))
            # Index is True if the assignment-pair results in an identical matrix
            assignment_satisfy_arr = np.sum(assignment_pair_satisfy_arr, axis=1) == self.num_species
            if IS_DEBUG:
                if np.sum(assignment_satisfy_arr) == 0:
                    import pdb; pdb.set_trace()
            return assignment_satisfy_arr, is_truncated  # Booleans indicating acceptable assignments
        #####
        #
        # Compare candidate assignments to the reference network
        if identity == cn.ID_WEAK:
            assignment_pair_arr, new_is_truncated = compare(participant=None)
        else:
            # Strong identity requires that both reactant and product satisfy the assignment
            reaction_assignment_pair_arr, reaction_is_truncated = compare(participant=cn.PR_REACTANT) 
            species_assignment_pair_arr, species_is_truncated = compare(participant=cn.PR_PRODUCT) 
            new_is_truncated = reaction_is_truncated or species_is_truncated
            assignment_pair_arr = np.logical_and(reaction_assignment_pair_arr, species_assignment_pair_arr)
        is_truncated = is_truncated or new_is_truncated
        # Construct the indices for reactions and species
        assignment_idxs = np.array(range(len(assignment_pair_arr)))
        num_reaction_assignment = reaction_assignment_arr.shape[0]
        # Calculate the indices for species and reaction assignments
        species_assignment_idxs = assignment_idxs[assignment_pair_arr]//num_reaction_assignment
        reaction_assignment_idxs = np.mod(assignment_idxs[assignment_pair_arr], num_reaction_assignment)
        assignment_pairs = []
        for species_idx, reaction_idx in zip(species_assignment_idxs, reaction_assignment_idxs):
            if species_assignment_arr.ndim == 2:
                species_assignment = species_assignment_arr[species_idx, :]
            else:
                species_assignment = species_assignment_arr
            true_species_assignment = species_assignment[inverse_permutation_pair.species_assignment]
            if reaction_assignment_arr.ndim == 2:
                reaction_assignment = reaction_assignment_arr[reaction_idx, :]
            else:
                reaction_assignment = reaction_assignment_arr
            true_reaction_assignment = reaction_assignment[inverse_permutation_pair.reaction_assignment]
            assignment_pair = AssignmentPair(species_assignment=true_species_assignment,
                                            reaction_assignment=true_reaction_assignment)
            assignment_pairs.append(assignment_pair)
        # Construct the result
        if IS_DEBUG:
            if len(assignment_pairs) == 0:
                import pdb; pdb.set_trace()
        return StructurallyIdenticalResult(assignment_pairs=assignment_pairs,
                is_truncated=is_truncated,
                species_compression_factor=species_assignment_result.compression_factor,
                reaction_compression_factor=reaction_assignment_result.compression_factor)
    
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
