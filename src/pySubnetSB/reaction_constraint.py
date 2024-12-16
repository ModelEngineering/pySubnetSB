'''Constraints for reactions.'''

"""
Enumerated constraints.
    Reaction type (equality)
    Number precdecessors of each type
    Number of successors of each type
"""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB.constraint import Constraint, NULL_NMAT # type: ignore

import numpy as np
from typing import List


#####################################
class ReactionConstraint(Constraint):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix,
                 is_subnet:bool=False):
        """
        Args:
            reactant_nmat (NamedMatrix)
            product_nmat (NamedMatrix)
        """
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat, is_subnet=is_subnet)
        self._is_initialized = False
        self._numerical_enumerated_nmat = NULL_NMAT  
        self._numerical_categorical_nmat = NULL_NMAT
        self._bitwise_enumerated_nmat = NULL_NMAT
        self._bitwise_categorical_nmat = NULL_NMAT
        self._one_step_nmat = NULL_NMAT
        self._to_reaction_arr = np.eye(self.num_reaction)

    def _initialize(self):
        if self._is_initialized:
            return
        self._numerical_enumerated_nmat = NamedMatrix.hstack([
              self.makeSuccessorPredecessorConstraintMatrix(),
              self.makeNStepConstraintMatrix(num_step=2),
        ])
        self._numerical_categorical_nmat = NamedMatrix.hstack([self._makeClassificationConstraintMatrix(),
            self._makeAutocatalysisConstraintMatrix()])
        self._is_initialized = True

    ################# OVERLOADED PARENT CLASS METHODS #################
    @property
    def row_names(self)->List[str]:
        return self.reactant_nmat.column_names
    
    @property
    def description(self)->str:
        return "reactions"

    @property
    def numerical_enumerated_nmat(self)->NamedMatrix:
        self._initialize()
        return self._numerical_enumerated_nmat
    
    @property
    def categorical_nmat(self)->NamedMatrix:
        self._initialize()
        return self._numerical_categorical_nmat

    @property
    def bitwise_enumerated_nmat(self)->NamedMatrix:
        return self._bitwise_enumerated_nmat
    
    @property
    def one_step_nmat(self)->NamedMatrix:
        """Calculates the successor matrix for the species monopartite graph.

        Returns:
            np.ndarray: _description_
        """
        # Create the monopartite graph
        if self._one_step_nmat is NULL_NMAT:
            incoming_arr = self.reactant_nmat.values.T
            outgoing_arr = self.product_nmat.values.T
            arr = np.sign(np.matmul(outgoing_arr, incoming_arr.T)).astype(int)
            self._one_step_nmat = NamedMatrix(arr, row_names=self.reactant_nmat.column_names,
                                row_description='reactions',
                                column_description='reactions',
                                column_names=self.reactant_nmat.column_names)
        return self._one_step_nmat
    
    @property
    def to_reaction_arr(self)->np.ndarray:
        return self._to_reaction_arr
    
    ##################################

    def __repr__(self)->str:
        return "Reaction--" + super().__repr__()
    
    def _makeClassificationConstraintMatrix(self)->NamedMatrix:
        """Make constraints for the reaction type. These are the number of
        reactions of each type.

        Returns:
            NamedMatrix: Rows are reactions, columns are constraints by count of reaction type.
        """
        array = np.array([self.reaction_classification_arr[n].encoding for n in range(self.num_reaction)])
        array = np.reshape(array, (len(array), 1)).astype(int)
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
        autocatalysis_arr = np.reshape(autocatalysis_arr, (len(autocatalysis_arr), 1)).astype(int)
        column_names = ["autocatalysis"]
        named_matrix = NamedMatrix(autocatalysis_arr, row_names=self.reactant_nmat.column_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description="constraints",
                           column_names=column_names)
        return named_matrix