'''Constraint species.'''

"""
Enumerated constraints.
   Number of occurrence of the species as a reactant in the different reaction types.
   Number of predecessors in the inferred monopartite graph
   Number of successors in the inferred monopartite graph
No categorical constraints.
"""

import sirn.constants as cn # type: ignore
from sirn.named_nmatrix import NamedMatrix # type: ignore
from sirn.constraint import Constraint, ReactionClassification # type: ignore
import sirn.util as util # type: ignore

import numpy as np
from typing import Optional, List


#####################################
class _ReactionClassificationConstraint(object):
    """A constraint for a reaction classification."""

    def __init__(self, reaction_classification:ReactionClassification, is_reactant:bool):
        self.reaction_classification = reaction_classification
        self.is_reaction = is_reactant
        self.count = 0

    def add(self, value:int=1):
        self.count += value

    def __repr__(self)->str:
        prefix = "R" if self.is_reaction else "P"
        return f"{prefix}_{self.reaction_classification}={self.count}"
    
    @classmethod
    def getReactionClassificationConstraints(cls)->dict:
        """Gets a dictionary of reaction classification constraints.
        
        Returns:
            dict[str, ReactionClassificationConstraint]: Key is the reaction classification
        """
        reaction_classifications = ReactionClassification.getReactionClassifications()
        constraints = {}
        for reaction_classification in reaction_classifications:
            constraints[str(reaction_classification)] = cls(reaction_classification, True)
            constraints[str(reaction_classification)] = cls(reaction_classification, False)
        constraint_dct = {str(reaction_classification): cls(reaction_classification, True)
                            for reaction_classification in reaction_classifications}
        return constraint_dct


#####################################
class SpeciesConstraint(Constraint):

    def __repr__(self)->str:
        return "Species--" + super().__repr__()
    
    def _makeReactionClassificationConstraints(self)->NamedMatrix:
        """Make constraints for the reaction classification

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        constraint_dct = _ReactionClassificationConstraint.getReactionClassificationConstraints()
        arrays = []
        for i_species in range(self.num_species):
            array = np.zeros(len(constraint_dct.keys()))
            for idx, constraint_str in enumerate(constraint_dct.keys()):
                for i_reaction in range(self.num_reaction):
                    if str(self.reaction_classifications[i_reaction]) == constraint_str:
                        if self.reactant_nmat.values[i_species, i_reaction] > 0:
                            array[idx] += 1
                        if self.reactant_nmat.values[i_species, i_reaction] > 0:
                            array[idx] += 1
            arrays.append(array)
        # Make the NamedMatrix
        column_names = list(constraint_dct.keys())
        return NamedMatrix(np.array(arrays), row_names=self.reactant_nmat.row_names,
                           row_description=self.reactant_nmat.column_description,
                           column_description=self.reactant_nmat.row_description,
                           column_names=column_names)
    
    
    def _makeSpeciesMonopartiteConstraints(self)->NamedMatrix:
        """Make constraints for the species monopartite graph.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        # Create the monopartite graph
        monopartite_arr = self.reactant_nmat.values*np.transpose(self.product_nmat.values)
        # Count predecessor species
        precessor_arr = np.sum(monopartite_arr, axis=1)
        # Count successor species
        successor_arr = np.sum(monopartite_arr, axis=0)
        # Construct the NamedMatrix
        array = np.hstack([precessor_arr, successor_arr])
        column_names = ["predecessors", "successors"]
        return NamedMatrix(array, row_names=self.reactant_nmat.row_names,
                           row_description=self.reactant_nmat.row_description,
                           column_description="constraints",
                           column_names=column_names)
    
    def _makeSpeciesConstraintMatrix(self)->NamedMatrix:
        """Make the species constraint matrix.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        return NamedMatrix.hstack([self._makeReactionClassificationConstraints(),
                                   self._makeSpeciesMonopartiteConstraints()])