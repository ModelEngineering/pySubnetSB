'''Constraints for reactions.'''

"""
Enumerated constraints.
   Number of occurrence of the species as a reactant in the different reaction types.
   Number of predecessors in the inferred monopartite graph
   Number of successors in the inferred monopartite graph
No categorical constraints.
"""

import sirn.constants as cn # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore
from sirn.constraint import Constraint, ReactionClassification, NULL_NMAT # type: ignore
import sirn.util as util # type: ignore

import numpy as np
from typing import Optional, List


#####################################
class SpeciesConstraint(Constraint):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix, is_subset:bool=False):
        """
        Args:
            reactant_nmat (NamedMatrix)
            product_nmat (NamedMatrix)
            is_subset (bool, optional) Consider self as a subset of other.
        """
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self.is_subset = is_subset
        self.is_initialized = False
        self._equality_nmat = NULL_NMAT
        self._inequality_nmat = NULL_NMAT

    @property
    def equality_nmat(self)->NamedMatrix:
        if not self.is_initialized:
            if self.is_subset:
                self._inequality_nmat = self._makeSpeciesConstraintMatrix()
            else:
                self._equality_nmat = self._makeSpeciesConstraintMatrix()
            self.is_initialized = True
        return self._equality_nmat
        
    
    @property
    def inequality_nmat(self)->NamedMatrix:
        if not self.is_initialized:
            if self.is_subset:
                self._inequality_nmat = self._makeSpeciesConstraintMatrix()
            else:
                self._equality_nmat = self._makeSpeciesConstraintMatrix()
            self.is_initialized = True
        return self._inequality_nmat

    def __repr__(self)->str:
        return "Species--" + super().__repr__()
    
    def _makeReactionClassificationConstraints(self)->NamedMatrix:
        """Make constraints for the reaction classification. These are {R, P} X {ReactionClassifications}
        where R is reactant and P is product.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        reaction_classifications = [str(c) for c in ReactionClassification.getReactionClassifications()]
        reactant_arrays = []
        product_arrays = []
        for i_species in range(self.num_species):
            reactant_array = np.zeros(len(reaction_classifications))
            product_array = np.zeros(len(reaction_classifications))
            for i_reaction in range(self.num_reaction):
                i_constraint_str = str(self.reaction_classifications[i_reaction])
                idx = reaction_classifications.index(i_constraint_str)
                if self.reactant_nmat.values[i_species, i_reaction] > 0:
                    reactant_array[idx] += 1
                if self.product_nmat.values[i_species, i_reaction] > 0:
                    product_array[idx] += 1
            reactant_arrays.append(reactant_array)
            product_arrays.append(product_array)
        # Construct full array and labels
        arrays = np.concatenate([reactant_arrays, product_arrays], axis=1)
        reactant_labels = [f"r_{c}" for c in reaction_classifications]
        product_labels = [f"p_{c}" for c in reaction_classifications]
        column_labels = reactant_labels + product_labels
        # Make the NamedMatrix
        named_matrix = NamedMatrix(np.array(arrays), row_names=self.reactant_nmat.row_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description=self.reactant_nmat.row_description,
                           column_names=column_labels)
        return named_matrix
    
    def _makeSpeciesMonopartiteConstraints(self)->NamedMatrix:
        """Make constraints for the species monopartite graph. These are the number of
        predecessor and successor nodes.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        # Create the monopartite graph
        incoming_arr = self.reactant_nmat.values
        outgoing_arr = self.product_nmat.values
        monopartite_arr = np.sign(np.matmul(incoming_arr, outgoing_arr.T))
        # Count predecessor species
        predecessor_arr = np.sum(monopartite_arr, axis=0)
        # Count successor species
        successor_arr = np.sum(monopartite_arr, axis=1)
        import pdb; pdb.set_trace()
        # Construct the NamedMatrix
        array = np.vstack([predecessor_arr, successor_arr]).T
        column_names = ["num_predecessor", "num_successor"]
        named_matrix = NamedMatrix(array, row_names=self.reactant_nmat.row_names,
                           row_description=self.reactant_nmat.row_description,
                           column_description="constraints",
                           column_names=column_names)
        return named_matrix
    
    def _makeSpeciesConstraintMatrix(self)->NamedMatrix:
        """Make the species constraint matrix.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        return NamedMatrix.hstack([self._makeReactionClassificationConstraints(),
                                   self._makeSpeciesMonopartiteConstraints()])