'''Constraint objects for subgraphs.'''

"""
Constraint objects have two kinds of constraints:
   1. Categorical constraints that must have an exact match.
   2. Enumerated constaints where the subgraph must have a value no greater than its target.

Constraints are NamedMatrices such that the columns are constraints and the rows are instances.
For reaction networks, instances are either species or reactions.
"""

import sirn.constants as cn # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import json
import numpy as np
from typing import Optional, List


#####################################
class ReactionClassification(object):
    MAX_REACTANT = 3
    MAX_PRODUCT = 3

    def __init__(self, num_reactant:int, num_product:int):
        self.num_reactant = num_reactant
        self.num_product = num_product

    def __repr__(self)->str:
        labels = ["null", "uni", "bi", "multi"]
        return f"{labels[self.num_reactant]}-{labels[self.num_product]}"

    @classmethod 
    def getReactionClassifications(cls)->List['ReactionClassification']:
        """Gets a list of reaction classifications."""
        classifications = []
        for num_reactant in range(ReactionClassification.MAX_REACTANT):
            for num_product in range(ReactionClassification.MAX_PRODUCT):
                classifications.append(cls(num_reactant, num_product))
        return classifications


#####################################
class Constraint(object):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix):
        """
        Args:
            criteria (np.array): A vector of criteria.
        """
        self.reactant_nmat = reactant_nmat
        self.product_nmat = product_nmat
        # Calculated
        self.num_species, self.num_reaction = self.reactant_nmat.num_row, self.reactant_nmat.num_column
        self.reaction_classifications = self.classifyReactions()
        # Output
        self.categorical_constraint:Optional[NamedMatrix] = None
        self.enumerated_constraint:Optional[NamedMatrix] = None

    def __repr__(self)->str:
        return str(self.reactant_nmat, self.product_nmat)

    def __eq__(self, other)->bool:
        if not isinstance(other, self.__class__):
            return False
        if self.reactant_nmat != other.reactant_nmat:
            return False
        if self.product_nmat != other.product_nmat:
            return False
        return True

    def copy(self):
        return self.__class__(self.reactant_nmat.copy(), self.product_nmat.copy())
    
    def serialize(self)->str:
        """Serializes the boundary values."""
        return json.dumps({cn.S_ID: str(self.__class__), cn.S_REACTANT_NMAT: self.reactant_nmat.serialize(),
                           cn.S_PRODUCT_NMAT: self.product_nmat.serialize()})

    @classmethod
    def deserialize(cls, string:str)->'Constraint':
        """Deserializes the boundary values."""
        dct = json.loads(string)
        if not str(cls) in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        reactant_nmat = NamedMatrix.deserialize(dct[cn.S_REACTANT_NMAT])
        product_nmat = NamedMatrix.deserialize(dct[cn.S_PRODUCT_NMAT])
        return cls(reactant_nmat, product_nmat)
    
    def classifyReactions(self)->List[ReactionClassification]:
        """Classify the reactions based on the number of reactants and products.

        Returns:
            List[ReactionClassification]
        """
        classifications = []
        for idx in range(self.num_reaction):
            num_reactant = np.sum(self.reactant_nmat.values[idx, :])
            num_product = np.sum(self.product_nmat.values[idx, :])
            classifications.append(ReactionClassification(num_reactant, num_product))
        return classifications