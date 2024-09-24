'''Constraints for reactions.'''

"""
Enumerated constraints.
    Reaction type (equality)
    Number precdecessors of each type
    Number of successors of each type
"""

import sirn.constants as cn # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore
from sirn.constraint import Constraint, NULL_NMAT # type: ignore

import numpy as np


#####################################
class ReactionConstraint(Constraint):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix, is_subset:bool=False):
        """
        Args:
            reactant_nmat (NamedMatrix)
            product_nmat (NamedMatrix)
            is_subset (bool, optional) Consider self as a subset of other.
        """
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self._is_initialized = False
        self._enumerated_nmat = NULL_NMAT  
        self._categorical_nmat = NULL_NMAT

    def _initialize(self):
        if self._is_initialized:
            return
        self._enumerated_nmat = self._makeSuccessorConstraintMatrix()
        self._categorical_nmat = NamedMatrix.hstack([self._makeClassificationConstraintMatrix(),
            self._makeAutocatalysisConstraintMatrix()])
        self._is_initialized = True

    @property
    def enumerated_nmat(self)->NamedMatrix:
        self._initialize()
        return self._enumerated_nmat
    
    @property
    def categorical_nmat(self)->NamedMatrix:
        self._initialize()
        return self._categorical_nmat

    def __repr__(self)->str:
        return "Reaction--" + super().__repr__()
    
    def _makeSuccessorConstraintMatrix(self)->NamedMatrix:
        """Make constraints for the reaction monopartite graph. These are the number of
        successor reactions of each type.

        Returns:
            NamedMatrix: Rows are reactions, columns are constraints by count of reaction type.
              <ReactionType>
        """
        # Create the monopartite graph
        incoming_arr = self.reactant_nmat.values.T
        outgoing_arr = self.product_nmat.values.T
        monopartite_arr = np.sign(np.matmul(outgoing_arr, incoming_arr.T))
        # Process the sucessors
        successor_arrays:list = []
        for ifrom in range(self.num_reaction):
            # Process the from row
            successor_dct = {str(n): 0 for n in self._reaction_classes}
            for ito in range(self.num_reaction):
                successor_dct[str(self.reaction_classification_arr[ito])] += monopartite_arr[ifrom, ito]
            array = list(successor_dct.values())
            successor_arrays.append(array)
        successor_arr = np.array(successor_arrays)
        # Validate the result
#        for irow in range(self.num_reaction):
#            if np.sum(successor_arr[irow, :]) != np.sum(monopartite_arr[irow, :]):
#                import pdb; pdb.set_trace()
        # Make the NamedMatrix
        named_matrix = NamedMatrix(successor_arr, row_names=self.reactant_nmat.column_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description="constraints",
                           column_names=self._reaction_classes)
        return named_matrix
    
    def _makeClassificationConstraintMatrix(self)->NamedMatrix:
        """Make constraints for the reaction type. These are the number of
        reactions of each type.

        Returns:
            NamedMatrix: Rows are reactions, columns are constraints by count of reaction type.
        """
        array = np.array([str(self.reaction_classification_arr[n]) for n in range(self.num_reaction)])
        array = np.reshape(array, (len(array), 1))
        column_names = ["reaction_classifications"]
        named_matrix = NamedMatrix(array, row_names=self.reactant_nmat.column_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description="constraints",
                           column_names=column_names)
        return named_matrix
    
    def _makeAutocatalysisConstraintMatrix(self)->NamedMatrix:
        """Make constraints for autocatalysis. These are the number of
        reactions of each type that are autocatalytic.

        Returns:
            NamedMatrix: Rows are reactions, columns are constraints by count of reaction type.
        """
        autocatalysis_arr = np.sum(self.reactant_nmat.values * self.product_nmat.values, axis=0)
        autocatalysis_arr = np.sign(autocatalysis_arr)
        autocatalysis_arr = np.reshape(autocatalysis_arr, (len(autocatalysis_arr), 1))
        column_names = ["autocatalysis"]
        named_matrix = NamedMatrix(autocatalysis_arr, row_names=self.reactant_nmat.column_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description="constraints",
                           column_names=column_names)
        return named_matrix