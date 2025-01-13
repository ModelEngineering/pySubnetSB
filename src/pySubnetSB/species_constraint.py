'''Constraint species.'''

"""
Implements the calculation of categorical and enumerated constraints for species.
  Enumerated constraints: 
    counts of reaction types in which a species is a reactant or product
    number of autocatalysis reactions for a species.
"""

from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB.constraint import Constraint, ReactionClassification, NULL_NMAT # type: ignore
import pySubnetSB.util as util  # type: ignore
from pySubnetSB.constraint_option_collection import SpeciesConstraintOptionCollection # type: ignore

import numpy as np
from typing import List



#####################################
class SpeciesConstraint(Constraint):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix, is_subnet:bool=True,
              species_constraint_option_collection:SpeciesConstraintOptionCollection=SpeciesConstraintOptionCollection()):  
        """
        Args:
            reactant_nmat (NamedMatrix)
            product_nmat (NamedMatrix)
            is_subnet (bool, optional) Consider self as a subset of other.
        """
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat, is_subnet=is_subnet)
        #
        self._is_initialized = False
        self._numerical_enumerated_nmat = NULL_NMAT
        self._categorical_nmat = NULL_NMAT
        self._bitwise_enumerated_nmat = NULL_NMAT
        self._one_step_nmat = NULL_NMAT
        self.species_constraint_options = species_constraint_option_collection

    ################# OVERLOADED PARENT CLASS METHODS #################
    @property
    def row_names(self)->List[str]:
        return self.reactant_nmat.row_names
    
    @property
    def description(self)->str:
        return "species"

    @property
    def numerical_enumerated_nmat(self)->NamedMatrix:
        if not self._is_initialized:
            matrices = []
            if self.species_constraint_options.isTrue("is_reactant_product_count_constraint_matrix"):
                matrices.append(self._makeReactantProductCountConstraintMatrix())
            if self.species_constraint_options.isTrue("is_autocatalysis_constraint"):
                matrices.append(self._makeAutocatalysisConstraint())
            if self.species_constraint_options.isTrue("is_reactant_product_constraint_matrix"):
                matrices.append(self._makeReactantProductConstraintMatrix())
            if self.species_constraint_options.isTrue("is_successor_predecessor_constraint_matrix"):
                matrices.append(self.makeSuccessorPredecessorConstraintMatrix())
            if self.species_constraint_options.isTrue("is_n_step_constraint_matrix"):
                matrices.append(self.makeNStepConstraintMatrix(num_step=2))
            #
            if len(matrices) == 0:
                self._numerical_enumerated_nmat = NULL_NMAT
            else:
                self._numerical_enumerated_nmat = NamedMatrix.hstack(matrices)
            self._is_initialized = True
        return self._numerical_enumerated_nmat

    @property
    def categorical_nmat(self)->NamedMatrix:
        if self._categorical_nmat is NULL_NMAT:
              #self._categorical_mat = self._makeAutocatalysisConstraint()  # Categorical version
              pass
        return self._categorical_nmat

    @property
    def bitwise_enumerated_nmat(self)->NamedMatrix:
        if self._bitwise_enumerated_nmat is NULL_NMAT:
            #self._bitwise_enumerated_nmat = self._makeBitwiseReactantProductConstraintMatrix()
            pass
        return self._bitwise_enumerated_nmat
    
    @property
    def one_step_nmat(self)->NamedMatrix:
        """Calculates the successor matrix for the species monopartite graph.

        Returns:
            np.ndarray: _description_
        """
        # Create the monopartite graph
        if self._one_step_nmat is NULL_NMAT:
            incoming_arr = self.reactant_nmat.values
            outgoing_arr = self.product_nmat.values
            arr = np.sign(np.matmul(incoming_arr, outgoing_arr.T)).astype(int)
            self._one_step_nmat = NamedMatrix(arr, row_names=self.reactant_nmat.row_names,
                                row_description='species',
                                column_description='species',
                                column_names=self.reactant_nmat.row_names)
        return self._one_step_nmat
    
    @property
    def to_reaction_arr(self)->np.ndarray:
        return self.reactant_nmat.values
    
    ##################################

    def __repr__(self)->str:
        return "Species--" + super().__repr__()
    
    def _makeReactantProductConstraintMatrix(self)->NamedMatrix:
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
                i_constraint_str = str(self.reaction_classification_arr[i_reaction])
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
                           row_description='species',
                           column_description='constraints',
                           column_names=column_labels)
        return named_matrix
    
    def _makeReactantProductCountConstraintMatrix(self)->NamedMatrix:
        """Calculates the count of reactions in which a species is a reactant or product.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        reactant_count_arr = np.sum(self.reactant_nmat.values, axis=1)
        product_count_arr = np.sum(self.product_nmat.values, axis=1)
        array = np.array([reactant_count_arr, product_count_arr]).T
        column_labels = ["reactant_count", "product_count"]
        # Make the NamedMatrix
        named_matrix = NamedMatrix(array, row_names=self.reactant_nmat.row_names,
                           row_description='species',
                           column_description='constraints',
                           column_names=column_labels)
        return named_matrix
    
    def _makeBitwiseReactantProductConstraintMatrix(self)->NamedMatrix:
        """For each reach, construct two bit vectors, one for reactants and one for products,
        that species the type of reactions in which the species is present (as a reactant or product).

        Returns:
            NamedMatrix: Rows are species, columns are bit vectors for reactions where species is a reactant or product.
        """
        reactant_bit_vectors:List[int] = []
        product_bit_vectors:List[int] = []
        for i_species in range(self.num_species):
            reactant_reaction_type_bit = 0
            product_reaction_type_bit = 0
            for i_reaction in range(self.num_reaction):
                index = self.reaction_classification_arr[i_reaction].index
                if self.reactant_nmat.values[i_species, i_reaction] > 0:
                    reactant_reaction_type_bit |= 1 << index
                if self.product_nmat.values[i_species, i_reaction] > 0:
                    product_reaction_type_bit |= 1 << index
            reactant_bit_vectors.append(reactant_reaction_type_bit)
            product_bit_vectors.append(product_reaction_type_bit)
        # Construct full array and labels
        array = np.array([reactant_bit_vectors, product_bit_vectors]).T
        column_labels = ["reactant_bit_vector", "product_bit_vector"]
        # Make the NamedMatrix
        named_matrix = NamedMatrix(array, row_names=self.reactant_nmat.row_names,
                           row_description='species',
                           column_description='constraints',
                           column_names=column_labels)
        return named_matrix
    
    def _makeAutocatalysisConstraint(self)->NamedMatrix:
        """Indicates if the species is involved in an autocatalysis reaction.

        Returns:
            NamedMatrix. is_autocatalysis
        """
        column_names =  ['is_autocatalysis']
        array = self.reactant_nmat.values * self.product_nmat.values > 0
        vector = np.sum(array, axis=1)
        vector = np.reshape(vector, (len(vector), 1)).astype(int)
        #vector = np.sign(vector)
        named_matrix = NamedMatrix(vector, row_names=self.reactant_nmat.row_names,
                           row_description='species',
                           column_description='constraints',
                           column_names=column_names)
        return named_matrix