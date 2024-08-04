'''Abstraction for a reaction network. This is represented by a reactant PMatrix and product PMatrix.'''

"""
Subnets.

1. Don't create an expanded stoichiometry matrix. Instead, have two matrices for reactants and products.
2. Use row encodings for species and column encodings for reactions.
"""

from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn.pmatrix import PMatrix  # type: ignore
from sirn.util import hashArray  # type: ignore
from sirn import constants as cn  # type: ignore

import collections
import itertools
import os
import numpy as np
from typing import Optional, Union, List, Tuple


StrongStructuralIdentityResult = collections.namedtuple('StrongStructuralIdentityResult',
        ['row_perm', 'column_perm', 'num_perm']) 
StackedCriteriaCountMatrices = collections.namedtuple('StackedCriteriaCountMatrices',
        ['self_adjacent_pair_species_mat',
         'other_all_pair_species_mat',
         'self_adjacent_pair_reaction_mat',
         'other_all_pair_reaction_mat'])


class StructurallyIdenticalResult(object):
    # Auxiliary object returned by isStructurallyIdentical

    def __init__(self,
                 is_compatible:bool=False,
                 is_structural_identity_weak:bool=False,
                 is_structural_identity_strong:bool=False,
                 is_excessive_perm:bool=False,
                 num_perm:int=0,
                 this_row_perm:Optional[np.ndarray[int]]=None,
                 this_column_perm:Optional[np.ndarray[int]]=None,
                 other_row_perm:Optional[np.ndarray[int]]=None,
                 other_column_perm:Optional[np.ndarray[int]]=None)->None:
        """
        Args:
            is_compatible (bool): has the same shapes and row/column encodings
            is_structural_identity_weak (bool)
            is_structural_identity_strong (bool)
            is_excessive_perm (bool): the number of permutations exceeds a threshold
            num_perm (int): number of permutations
            this_row_perm (np.ndarray[int]): permutation for the rows to make the matrices equal
            this_column_perm (np.ndarray[int]): column permutations to make the matrices equal
            other_row_perm (np.ndarray[int]): row permutation of other to make the matrices equal
            other_column_perm (np.ndarray[int]): column permutations of other to make the matrices equal
        """
        self.is_excessive_perm = is_excessive_perm
        self.is_compatible = is_compatible
        self.is_structural_identity_weak = is_structural_identity_weak
        self.is_structural_identity_strong = is_structural_identity_strong
        self.num_perm = num_perm
        self.this_row_perm = this_row_perm
        self.this_column_perm = this_column_perm
        self.other_row_perm = other_row_perm
        self.other_column_perm = other_column_perm
        #
        if self.is_structural_identity_strong:
            self.is_structural_identity_weak = True
        if self.is_structural_identity_weak:
            self.is_compatible = True
        if is_excessive_perm:
            self.is_structural_identity_weak = False
            self.is_structural_identity_strong = False


class StructurallyIdenticalSubsetResult(object):

    def __init__(self, structurally_identical_results:List[StructurallyIdenticalResult],
                 num_perm:int)->None:
        """Describes results from a search for a structurally identifiable subset.

        Args:
            structurally_identical_results (List[StructurallyIdenticalResult]): _description_
            num_perm (int, optional): _description_. Defaults to 0.
        """
        self.structurally_identical_results = structurally_identical_results
        self.num_perm = num_perm
    
    @property
    def is_excessive_perm(self)->bool:
        if len(self.structurally_identical_results) == 0:
            return False
        return all([v.is_excessive_perm for v in self.structurally_identical_results])
    
    @property
    def is_compatible(self)->bool:
        if len(self.structurally_identical_results) == 0:
            return False
        return any([v.is_compatible for v in self.structurally_identical_results])
    
    @property
    def is_structural_identity_weak(self)->bool:
        if len(self.structurally_identical_results) == 0:
            return False
        return any([v.is_structural_identity_weak for v in self.structurally_identical_results])
    
    @property
    def is_structural_identity_strong(self)->bool:
        if len(self.structurally_identical_results) == 0:
            return False
        return any([v.is_structural_identity_strong for v in self.structurally_identical_results])


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
        # Reactant stoichiometry matrix is negative
        self.reactant_pmatrix = self._makePMatrix(reactant_mat, multiplier=-1.0)
        self.product_pmatrix = self._makePMatrix(product_mat)
        #
        if network_name is None:
            network_name = str(np.random.randint(0, 10000000))
        self.network_name = network_name
        stoichiometry_array = self.product_pmatrix.array + self.reactant_pmatrix.array
        self.stoichiometry_pmatrix = PMatrix(stoichiometry_array)
        # Hash values for simple stoichiometry (only stoichiometry matrix) and non-simple stoichiometry
        self.nonsimple_hash = hashArray(np.array([self.reactant_pmatrix.hash_val,
                                                  self.product_pmatrix.hash_val]))
        self.simple_hash = self.stoichiometry_pmatrix.hash_val

    def _makePMatrix(self, matrix:Union[np.ndarray, PMatrix], multiplier:float=1.0)->PMatrix:
        """
        Handles having a matrix or a PMatrix.

        Args:
            matrix (Union[np.ndarray, PMatrix])
            multiplier (float, optional): Multiplier for the matrix. Defaults to 1.0.

        Returns:
            PMatrix
        """
        if isinstance(matrix, PMatrix):
            row_names = matrix.row_names
            column_names = matrix.column_names
            matrix = matrix.array
        else:
            row_names = None
            column_names = None
        return PMatrix(matrix, row_names=row_names, column_names=column_names, multiplier=multiplier)

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

    def _isStrongStructurallyIdentical(self, other_array:np.ndarray,
            this_row_perms:List[np.ndarray[int]], this_column_perms:List[np.ndarray[int]],
             max_num_perm:int=cn.MAX_NUM_PERM)->StrongStructuralIdentityResult:
        """
        Does the checking necessary to determine strong structural identity of the product matrices.

        Args:
            other (Network)
            this_row_perms (List[np.ndarray[int]]): Permutations for this network
            this_column_perms (List[np.ndarray[int]]): Permutations for this network
            other_row_perm (np.ndarray[int]): Permutation for the other network
            other_column_perm (np.ndarray[int]): Permutation for the other network
            max_num_perm (int, optional): Maximum number of permutations to consider.

        Returns:
            StructurallyIdenticalResult
        """
        # Look at each permutation that makes the stoichiometry matrices equal
        num_perm = 0
        for this_row_perm, this_column_perm in zip(this_row_perms, this_column_perms):
            if num_perm >= max_num_perm:
                strong_structural_identity_result = StrongStructuralIdentityResult(
                    row_perm=None, column_perm=None, num_perm=num_perm)    
                return strong_structural_identity_result
            this_array = PMatrix.permuteArray(self.product_pmatrix.array,
                    this_row_perm, this_column_perm)
            num_perm += 1
            if np.allclose(this_array, other_array):
                strong_structural_identity_result = StrongStructuralIdentityResult(
                    row_perm=this_row_perm, column_perm=this_column_perm, num_perm=num_perm)    
                return strong_structural_identity_result
        strong_structural_identity_result = StrongStructuralIdentityResult(
            row_perm=None, column_perm=None, num_perm=num_perm)    
        return strong_structural_identity_result
    
    def isStructurallyIdentical(self, other:'Network', is_report:bool=False,
            max_num_perm:int=cn.MAX_NUM_PERM, is_sirn:bool=True,
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
            is_sirn (bool, optional): Use the SIRN algorithm. Defaults to True.
            is_structural_identity_type_weak (bool, optional): _description_. Defaults to False.

        Returns:
            StructurallyIdenticalResult
        """
        # Check for weak structural identity
        weak_identity = self.stoichiometry_pmatrix.isPermutablyIdentical(
            other.stoichiometry_pmatrix, is_find_all_perms=False, max_num_perm=max_num_perm,
            is_sirn=is_sirn)
        num_perm = weak_identity.num_perm
        if num_perm >= max_num_perm:
            return StructurallyIdenticalResult(num_perm=num_perm, is_excessive_perm=True)
        if not weak_identity:
            return StructurallyIdenticalResult(is_compatible=weak_identity.is_compatible, num_perm=num_perm)
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
            other.stoichiometry_pmatrix, is_find_all_perms=True, max_num_perm=revised_max_num_perm,
            is_sirn=is_sirn)
        num_perm += all_weak_identities.num_perm
        if num_perm >= max_num_perm:
            return StructurallyIdenticalResult(is_structural_identity_weak=True,
                    num_perm=num_perm, is_excessive_perm=True)
        if is_report:
            print(f'all_weak_identities: {len(all_weak_identities.this_column_perms)}')
        other_array = PMatrix.permuteArray(other.product_pmatrix.array,
                     all_weak_identities.other_row_perm,     # type: ignore
                     all_weak_identities.other_column_perm)  # type: ignore
        #
        strong_structurally_identical_result = self._isStrongStructurallyIdentical(other_array,
            all_weak_identities.this_row_perms, all_weak_identities.this_column_perms,
            max_num_perm=cn.MAX_NUM_PERM-num_perm)
        num_perm += strong_structurally_identical_result.num_perm
        if strong_structurally_identical_result.row_perm is None:
            # Wasn't able to match the product matrices
            return StructurallyIdenticalResult(is_structural_identity_weak=True,
                    num_perm=num_perm, this_row_perm=all_weak_identities.this_row_perms[0],
                    this_column_perm=all_weak_identities.this_column_perms[0],
                    other_row_perm=all_weak_identities.other_row_perm,
                    other_column_perm=all_weak_identities.other_column_perm)
        else:
            return StructurallyIdenticalResult(is_structural_identity_strong=True,
                    num_perm=num_perm, this_row_perm=strong_structurally_identical_result.row_perm,
                    this_column_perm=strong_structurally_identical_result.column_perm,
                    other_row_perm=all_weak_identities.other_row_perm,
                    other_column_perm=all_weak_identities.other_column_perm)

    def _makeStackedCriteriaCountMatrices(self, other:'Network')->StackedCriteriaCountMatrices:
        """
        Creates matrices that stack the criteria counts for reactant on top of products.

        Args:
            other (Network): _description_

        Raises:
            ValueError: _description_

        Returns:
            StackedCriteriaCountMatrices
        """
        self_adjacent_pair_species_mat = np.vstack([
              self.reactant_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array,
              self.product_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array])
        #
        self_adjacent_pair_reaction_mat = np.vstack([
              self.reactant_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array,
              self.product_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array])
        #
        other_all_pair_species_mat = np.vstack([
              self.reactant_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array,
              self.product_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array])
        #
        other_all_pair_reaction_mat = np.vstack([
              self.reactant_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array,
              self.product_pmatrix.row_collection.encoding.adjacent_pair_encoding_nm.array])
        return StackedCriteriaCountMatrices(
              self_adjacent_pair_species_mat=self_adjacent_pair_species_mat,
              self_adjacent_pair_reaction_mat=self_adjacent_pair_reaction_mat,
              other_all_pair_species_mat=other_all_pair_species_mat,
              other_all_pair_reaction_mat=other_all_pair_reaction_mat)
        
    # FIXME: Should I iterte across subsets here, constructing subnetworks, rath
    # FIXME: (1) validate the iterators; (2) combine their construction into a function; (3) draw a picture
    # FIXME: Subset iterator should be NetworkSbusetIterator since it requires knowledge of the networks.
    #        Both species and reactions must be compatible across both reactants and products
    #        I'm comparing criteria count matrices, which are encoded arrays.
    #        Comparisons are done for single arrays and pairs of arrays. .   
    def isStructurallyIdenticalSubset(self, other:'Network', is_report:bool=False,
          max_num_perm:int=cn.MAX_NUM_PERM,
          #is_structural_identity_weak:bool=False)->StructurallyIdenticalSubsetResult:
          is_structural_identity_weak:bool=False):
        """
        Determines if the current network is structurally identical to a subnetwork of other.
        The notions of weak and strong structural identity apply as before. The approach is to
        iterate across possible subnetworks of other to look for structural identity. A subnetwork
        is specified by a subset of the reactions of the network. These are identified based on
        the reactants and product of the reactions. We do this by forming a matrix that is
        the vertical concatenation of the negative of the reactant matrix and the product matrix. This
        is referred to as the network definition matrix.

        Args:
            other (Network)
            max_num_perm (int, optional): Maximum number of permutations to consider.
            is_report (bool, optional): Report on analysis progress. Defaults to False.
            is_structural_identity_type_weak (bool, optional): _description_. Defaults to False.

        Returns:
            StructurallyIdenticalSubsetResult
        """
        # Construct the network definition matrices for other. Rows are species. Columns are reactions.
        stacked_criteria_count_matrices = self._makeStackedCriteriaCountMatrices(other)
        species_iter = stacked_criteria_count_matrices.self_adjacent_pair_species_pmat.row_collection.subsetIterator(
                stacked_criteria_count_matrices.other_all_pair_species_pmat.row_collection)
        # Iterate over subsets of reactions in other
        is_done = False
        species_subsets = list(species_iter)
        for species in species_subsets:
            if is_done:
                break
            reaction_iter = stacked_criteria_count_matrices.self_adjacent_pair_reaction_pmat.row_collection.subsetIterator(
                  stacked_criteria_count_matrices.other_all_pair_reaction_pmat.row_collection)
            reaction_subsets = list(reaction_iter)
            subnetwork_results: List[StructurallyIdenticalResult] = []
            num_perm = 0
            for column_subset in column_subsets:
                # Construct the subnetwork
                other_reactant_pmatrix = other.reactant_pmatrix.getSubMatrix(column_idxs=column_subset,
                      row_idxs=row_subset)
                other_product_pmatrix = other.product_pmatrix.getSubMatrix(column_idxs=column_subset,
                      row_idxs=row_subset)
                other_subnetwork = Network(other_reactant_pmatrix, other_product_pmatrix)
                import pdb; pdb.set_trace()
                structurally_identical_result = self.isStructurallyIdentical(other_subnetwork,
                        max_num_perm=max_num_perm-num_perm, is_report=is_report,
                        is_structural_identity_weak=is_structural_identity_weak)
                num_perm += max(1, structurally_identical_result.num_perm)
                subnetwork_results.append(structurally_identical_result)
                if num_perm >= max_num_perm:
                    is_done = True
                    break
                if is_structural_identity_weak and structurally_identical_result.is_structural_identity_weak:
                    is_done = True
                    break
                if (not is_structural_identity_weak) and structurally_identical_result.is_structural_identity_strong:
                    is_done = True
                    break
            #
        return StructurallyIdenticalSubsetResult(subnetwork_results, num_perm)
    
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
    
    def subnetIterator(self, other:'Network'):
        """
        Iterates iterates over subnets of other that are compatible with self.

        Args:
            other: Network

        Yields:
            Network: subnet of other that is compatible with the current Network.
        """
        # Find compatible species. These are rows of the reactant and product matrices.
        self_row_collection = self.reactant_pmatrix.row_collection
        other_row_collection = other.reactant_pmatrix.row_collection
        reactant_candidates = self_row_collection.findCompatibleRows(
            self_row_collection.encoding.encoding_nm.matrix,
            other_row_collection.encoding.encoding_nm.matrix)
        self_column_collection = self.product_pmatrix.column_collection
        other_column_collection = other.product_pmatrix.column_collection
        product_candidates = self_column_collection.findCompatiblecolumns(
            self_column_collection.encoding.encoding_nm.matrix,
            other_column_collection.encoding.encoding_nm.matrix)
        species_candidates = [set(a).intersection(b) for a, b in zip(reactant_candidates, product_candidates)]
        # Find compatible species. These are rows of the reactant and product matrices.
        self_column_collection = self.reactant_pmatrix.column_collection
        other_column_collection = other.reactant_pmatrix.column_collection
        reactant_candidates = self_column_collection.findCompatiblecolumns(
            self_column_collection.encoding.encoding_nm.matrix,
            other_column_collection.encoding.encoding_nm.matrix)
        self_column_collection = self.product_pmatrix.column_collection
        other_column_collection = other.product_pmatrix.column_collection
        product_candidates = self_column_collection.findCompatiblecolumns(
            self_column_collection.encoding.encoding_nm.matrix,
            other_column_collection.encoding.encoding_nm.matrix)
        reaction_candidates = [set(a).intersection(b) for a, b in zip(reactant_candidates, product_candidates)]
        # Get the subsets
        species_iter = itertools.product(*species_candidates)
        species_subsets = list(species_iter)
        reaction_iter = itertools.product(*reaction_candidates)
        reaction_subsets = list(reaction_iter)
        for species_subset in species_subsets:
            # Check that subset is a permutation
            if len(set(species_subset)) < len(species_subset):
                continue
            # Handle pairwise constraints
            species_pairs = [(species_subset[i], species_subset[i+1]) for i in range(len(species_subset)-1)]
            other_sub_nm = other.encoding.all_pair_encoding_nm.getSubNamedMatrix(row_ids=pairs)
            # See if all pairwise constraints are satisfied
            if not np.all(self.encoding.adjacent_pair_encoding_nm.matrix <= other_sub_nm.matrix):
                continue
            # Process reactions
            for reaction_subset in reaction_subsets:
                # Check that subset is a permutation
                if len(set(reaction_subset)) < len(reaction_subset):
                    continue
                # Handle pairwise constraints
                reaction_pairs = [(reaction_subset[i], reaction_subset[i+1]) for i in range(len(reaction_subset)-1)]
                other_sub_nm = other.encoding.all_pair_encoding_nm.getSubNamedMatrix(row_ids=pairs)
                # See if all pairwise constraints are satisfied
                if not np.all(self.encoding.adjacent_pair_encoding_nm.matrix <= other_sub_nm.matrix):
                    continue
                other_reactant_pmatrix = other.reactant_pmatrix.getSubMatrix(column_idxs=reaction_subset,
                      row_idxs=species_subset)
                other_product_pmatrix = other.product_pmatrix.getSubMatrix(column_idxs=reaction_subset,
                      row_idxs=species_subset)
                other_subnetwork = Network(other_reactant_pmatrix, other_product_pmatrix)
                yield other_subnetwork