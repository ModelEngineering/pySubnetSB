'''Constraint objects for subgraphs.'''

"""
Constraint objects have two kinds of constraints:
   1. Equality constraints that must have an exact match.
   2. Inequality constaints where the reference must have a value no greater than its target.

Constraints are NamedMatrices such that the columns are constraints and the rows are instances.
For reaction networks, instances are either species or reactions.
"""

import sirn.constants as cn # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import json
import numpy as np
from typing import Optional, List

NULL_NMAT = NamedMatrix(np.array([[]]))
NULL_INT = -1


#####################################
class CompatibilityCollection(object):
    # A compatibility collection specifies the rows in self that are compatible with other.

    def __init__(self, num_self_row:int, num_other_row:int):
        self.num_self_row = num_self_row
        self.num_other_row = num_other_row
        self.compatiblibilities:list = [ [] for _ in range(num_self_row)]

    def add(self, reference_row:int, target_rows:List[int]):
        # Add rows in target that are compatible with a row in reference
        self.compatiblibilities[reference_row].extend(target_rows)

    def isFeasible(self)->bool:
        """Determines if there is at least one combination of compatible rows.

        Returns:
            bool
        """
        return all([len(row) > 0 for row in self.compatiblibilities])
    
    def makeFullCompatibility(self):
        """Makes a full compatibility collection."""
        full_compatibility = list(range(self.num_self_row))
        for irow in range(self.num_self_row):
            self.compatiblibilities[irow] = full_compatibility


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
            reactant_nmat: NamedMatrix
            product_nmat: NamedMatrix
            num_row: int
            num_column: int
        """
        self.reactant_nmat = reactant_nmat
        self.product_nmat = product_nmat
        self._num_row = NULL_INT
        # Calculated
        self.num_species, self.num_reaction = self.reactant_nmat.num_row, self.reactant_nmat.num_column
        self.reaction_classifications = self.classifyReactions()
        # Outputs are categorical_nmat and enumerated_nmat, which are implemented by subclass

    @property
    def num_row(self)->int:
        if self._num_row == NULL_INT:
            self._num_row = self.equality_nmat.num_row
        return self._num_row
    
    @property
    def equality_nmat(self)->NamedMatrix:
        # Columns are constraints for equality constraints
        raise NotImplementedError("_categorical_nmat be implemented by subclass.")
    
    @property
    def inequality_nmat(self)->NamedMatrix:
        # Columns are constraints for inequality constraints
        raise NotImplementedError("_enumerated_nmat be implemented by subclass.")

    def __repr__(self)->str:
        return str(self.reactant_nmat, self.product_nmat)

    def __eq__(self, other)->bool:
        if self.__class__.__name__ != other.__class__.__name__:
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
        return json.dumps({cn.S_ID: self.__class__.__name__,
                           cn.S_REACTANT_NMAT: self.reactant_nmat.serialize(),
                           cn.S_PRODUCT_NMAT: self.product_nmat.serialize(),
                           })

    @classmethod
    def deserialize(cls, string:str)->'Constraint':
        """Deserializes the boundary values."""
        dct = json.loads(string)
        if not cls.__name__ in dct[cn.S_ID]:
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

    @classmethod 
    def calculateCompatibilityVector(cls, self_constraint_nmat:NamedMatrix,
              other_constraint_nmat:NamedMatrix, is_equality:bool=True)->np.ndarray[bool]:  # type: ignore
        """
        Calculates the compatibility of the categorical constraints.

        Args:
            other: Constraint
            is_equality: bool (default: True) Equality or inequality

        Returns:
            np.ndarray[bool] - vector of booleans
                index n represents i*self_num_row + jth row in other
        """
        if self_constraint_nmat.num_column != other_constraint_nmat.num_column:
            raise ValueError("Incompatible number of columns.")
        #
        self_num_row = self_constraint_nmat.num_row
        other_num_row = other_constraint_nmat.num_row
        # Check for a null
        num_column = other_constraint_nmat.num_column
        # Calculate the CompatibilityCollection
        #    Create the self array with repeated blocks of each row
        self_arr = np.concatenate([self_constraint_nmat.values]*other_num_row, axis=1)
        self_arr = np.reshape(self_arr, (self_num_row*other_num_row, num_column))
        #    Create the other array with all values of each row
        other_arr = np.reshape(other_constraint_nmat.values, other_num_row*num_column)
        other_arr = np.concatenate([other_arr]*self_num_row)
        other_arr = np.reshape(other_arr, (self_num_row*other_num_row, num_column))
        # Calculate the compatibility boolean vector
        if is_equality:
            satisfy_arr = self_arr == other_arr
        else:
            satisfy_arr = self_arr <= other_arr
        return np.sum(satisfy_arr, axis=1) == num_column
    
    def makeCompatibilityCollection(self, other:'Constraint')->CompatibilityCollection:
        """
        Makes a collection of compatible constraints.
        """
        # Calculate the compatibility of the constraints
        if self.equality_nmat != NULL_NMAT:
            is_equality_compatibility = True
            equality_compatibility_arr = self.calculateCompatibilityVector(self.equality_nmat,
                  other.equality_nmat, is_equality=True)
        else:
            is_equality_compatibility = False
        if self.inequality_nmat != NULL_NMAT:
            is_inequality_compatibility = True
            inequality_compatibility_arr = self.calculateCompatibilityVector(self.inequality_nmat,
                  other.inequality_nmat, is_equality=False)
        else:
            is_inequality_compatibility = False
        # Calculate the compatibility vector
        if is_equality_compatibility and is_inequality_compatibility:
            compatibility_arr = equality_compatibility_arr & inequality_compatibility_arr
        elif is_equality_compatibility:
            compatibility_arr = equality_compatibility_arr
        elif is_inequality_compatibility:
            compatibility_arr = inequality_compatibility_arr
        else:
            raise ValueError("No compatibility constraints.")
        # Create the compatibility collection
        compatibility_collection = CompatibilityCollection(self.num_row, other.num_row)
        target_arr = np.array(range(other.num_row))
        for irow in range(self.num_row):
            #  Select the rows in other that are compatible with the row in self
            base_pos = irow*other.num_row
            idxs = range(base_pos, base_pos+other.num_row)
            sel_idxs = idxs[compatibility_arr[idxs]]
            target_rows = target_arr[compatibility_arr[sel_idxs]]
            compatibility_collection.add(irow, target_rows)
        return compatibility_collection