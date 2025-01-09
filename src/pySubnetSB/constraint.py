'''Constraint objects for subgraphs.'''

"""
Constraint objects have two kinds of constraints:
   1. Equality constraints that must have an exact match.
   2. Inequality constaints where the reference must have a value no greater than its target.

Constraints are NamedMatrices such that the columns are constraints and the rows are instances.
For reaction networks, instances are either species or reactions.

Subclasses are responsible for implementing:
  property: row_names - List[str] of row names
  property: description - str of description
  property: categorical_nmat - NamedMatrix of categorical constraints, those that are tested by equality
  property: numerical_enumerated_nmat - NamedMatrix of enumerated constraints
  property: bitwise_enumerated_nmat - NamedMatrix of boolean constraints such that the target
    must have a one bit for every bit in the reference.
  property: one_step_nmat - NamedMatrix of one step transitions
  property: to_reaction_arr - np.ndarray of reaction types for each reaction
"""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.named_matrix import NamedMatrix, NULL_NMAT # type: ignore
from pySubnetSB.compatibility_collection import CompatibilityCollection # type: ignore

import json
import scipy.special  # type: ignore
import numpy as np
import scipy  # type: ignore
import string  # type: ignore
from typing import List, Tuple

NULL_INT = -1
ALL = "all"
NONE = "none"


#####################################
class CompatibilityCollectionResult(object):

    def __init__(self, compatibility_collection:CompatibilityCollection,
          equality_compatibility_nmat=NULL_NMAT,
          inequality_compatibility_numerical_nmat=NULL_NMAT,
          inequality_compatibility_bitwise_nmat=NULL_NMAT,
          reference_equality_nmat:NamedMatrix=NULL_NMAT,
          target_equality_nmat:NamedMatrix=NULL_NMAT,
          reference_numerical_inequality_nmat:NamedMatrix=NULL_NMAT,
          target_numerical_inequality_nmat:NamedMatrix=NULL_NMAT,
          reference_bitwise_inequality_nmat:NamedMatrix=NULL_NMAT,
          target_bitwise_inequality_nmat:NamedMatrix=NULL_NMAT):
        self.compatibility_collection = compatibility_collection
        self.equality_compatibility_nmat = equality_compatibility_nmat
        self.inequality_compatibility_numerical_nmat = inequality_compatibility_numerical_nmat
        self.inequality_compatibility_bitwise_nmat = inequality_compatibility_bitwise_nmat
        self.reference_equality_nmat = reference_equality_nmat
        self.target_equality_nmat = target_equality_nmat
        self.reference_numerical_inequality_nmat = reference_numerical_inequality_nmat
        self.target_numerical_inequality_nmat = target_numerical_inequality_nmat
        self.reference_bitwise_inequality_nmat = reference_bitwise_inequality_nmat
        self.target_bitwise_inequality_nmat = target_bitwise_inequality_nmat


#####################################
class ReactionClassification(object):
    MAX_REACTANT = 3
    MAX_PRODUCT = 3
    MAX_ENCODING = 1000
    LABELS = ["null", "uni", "bi", "multi"]
    NUM_CLASSIFICATION = len(LABELS)**2
    REACTION_CLASSES:List['ReactionClassification'] = []
    REACTION_BITS = [1 << i for i in range(NUM_CLASSIFICATION)]

    def __init__(self, num_reactant:int, num_product:int):
        self.num_reactant = min(num_reactant, self.MAX_REACTANT)
        self.num_product = min(num_product, self.MAX_PRODUCT)
        self.encoding = self.num_reactant*100 + self.num_product
        self.index = self.getIndex(self.num_reactant, self.num_product)

    @classmethod
    def getIndex(cls, num_reactant, num_product):
        return int(num_reactant*len(cls.LABELS) + num_product)

    def __repr__(self)->str:
        result = f"{self.LABELS[int(self.num_reactant)]}-{self.LABELS[int(self.num_product)]}"
        return result

    @classmethod 
    def getReactionClassifications(cls)->List['ReactionClassification']:
        """Gets a list of reaction classifications."""
        if len(cls.REACTION_CLASSES) > 0:
            return cls.REACTION_CLASSES
        pair_arr = np.array([f"{cls.LABELS[n]}-{cls.LABELS[m]}" for n in range(cls.MAX_REACTANT+1)
              for m in range(cls.MAX_PRODUCT+1)])
        idx_arr = np.array([cls.getIndex(n, m) for n in range(cls.MAX_REACTANT+1) for m in range(cls.MAX_PRODUCT+1)])
        cls.REACTION_CLASSES = list(pair_arr[idx_arr])
        return cls.REACTION_CLASSES

    @classmethod 
    def makeReactionClassificationMatrix(cls, reaction_names:List[str],
          classifications:List['ReactionClassification'])->NamedMatrix:
        """Makes a matrix whose rows are reaction names and columns are classifications.

        Args:
            reaction_names: List[str]
            classifications: List[ReactionClassification]

        Returns:
            NamedMatrix: rows are reactions; columns are reaction classifications
        """
        zero_arr = np.zeros(cls.NUM_CLASSIFICATION)
        arrays = []
        for classification in classifications:
            arr = zero_arr.copy()
            arr[classification.index] = 1
            arrays.append(arr)
        result = NamedMatrix(np.array(arrays), row_names=reaction_names, column_names=cls.REACTION_CLASSES)
        return result
    

#####################################
class Constraint(object):

    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix, is_subnet:bool=True):
        """
        Args:
            reactant_nmat: NamedMatrix
            product_nmat: NamedMatrix
            num_row: int
            num_column: int
        """
        self.reactant_nmat = reactant_nmat
        self.product_nmat = product_nmat
        self.is_subnet = is_subnet
        # Calculated
        self._reaction_classes = ReactionClassification.getReactionClassifications()
        self.num_species, self.num_reaction = self.reactant_nmat.num_row, self.reactant_nmat.num_column
        self.reaction_classification_arr = self.classifyReactions()
        # Outputs are categorical_nmat and enumerated_nmat, which are implemented by subclass
        self._is_subnet_initialized = False
        self._equality_nmat = NULL_NMAT
        self._numerical_inequality_nmat = NULL_NMAT
        self._bitwise_inequality_nmat = NULL_NMAT

    @property
    def num_row(self)->int:
        return self.one_step_nmat.num_row

    @classmethod
    def calculateLog10UnconstrainedPermutation(cls, reference_size:int, target_size:int)->float:
        """Calculates the log10 of the number of permutations to examine if no constraint is applied.

        Args:
            reference_size: int (number of species and reactions)
            target_size: int (number of species and reactions)

        Returns:
            float
        """
        def calculatePermutation(num_reference:int, num_target:int)->int:
            result = scipy.special.factorial(num_target) /  \
                  scipy.special.factorial(num_target - num_reference, exact=True)
            return result
        log_permutation = np.log10(calculatePermutation(reference_size, target_size))
        return 2*log_permutation
    
    ################ SUBCLASS MUST IMPLEMENT ################
        
    @property
    def description(self)->str:
        raise NotImplementedError("self be implemented by subclass.")
        
    @property
    def row_names(self)->List[str]:
        raise NotImplementedError("row_names be implemented by subclass.")
        
    @property
    def categorical_nmat(self)->NamedMatrix:
        raise NotImplementedError("numerical_categorical_nmat be implemented by subclass.")
    
    @property
    def numerical_enumerated_nmat(self)->NamedMatrix:
        raise NotImplementedError("numerical_enumerated_nmat be implemented by subclass.")

    @property
    def bitwise_enumerated_nmat(self)->NamedMatrix:
        raise NotImplementedError("logical_enumerated_nmat be implemented by subclass.")
    
    @property
    def one_step_nmat(self)->NamedMatrix:
        raise NotImplementedError("one_step_nmat be implemented by subclass.")
    
    @property
    def to_reaction_arr(self)->np.ndarray:
        raise NotImplementedError("to_reaction_arr be implemented by subclass.")

    ########################################################
    
    def _initializeSubset(self):
        # Initialize the equality and inequality NamedMatrices
        if self._is_subnet_initialized:
            return
        if self.is_subnet:
            self._equality_nmat = self.categorical_nmat
            self._numerical_inequality_nmat = self.numerical_enumerated_nmat
            self._bitwise_inequality_nmat = self.bitwise_enumerated_nmat
        else:
            if self.categorical_nmat == NULL_NMAT:
                self._equality_nmat = self.numerical_enumerated_nmat
            elif self.numerical_enumerated_nmat == NULL_NMAT:
                self._equality_nmat = self.categorical_nmat
            else:
                self._equality_nmat = NamedMatrix.hstack([self.categorical_nmat, self.numerical_enumerated_nmat])
            if self.bitwise_enumerated_nmat != NULL_NMAT:
                self._equality_nmat = NamedMatrix.hstack([self._equality_nmat, self.bitwise_enumerated_nmat])
            self._numerical_inequality_nmat = NULL_NMAT
        self._is_subnet_initialized = True
   
    @property
    def equality_nmat(self)->NamedMatrix:
        self._initializeSubset()
        return self._equality_nmat
    
    @property
    def numerical_inequality_nmat(self)->NamedMatrix:
        self._initializeSubset()
        return self._numerical_inequality_nmat

    @property
    def bitwise_inequality_nmat(self)->NamedMatrix:
        self._initializeSubset()
        return self._bitwise_inequality_nmat

    def __repr__(self)->str:
        return "Categorical\n" + str(self.categorical_nmat) + "\n\nEnumerated\n" +  str(self.numerical_enumerated_nmat)

    def __eq__(self, other)->bool:
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        if self.reactant_nmat != other.reactant_nmat:
            return False
        if self.product_nmat != other.product_nmat:
            return False
        return True
    
    def setSubset(self, is_subnet:bool)->None:
        self.is_subnet = is_subnet
        self._is_subnet_initialized = False

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
            num_reactant = np.sum(self.reactant_nmat.values[:, idx])
            num_product = np.sum(self.product_nmat.values[:, idx])
            classifications.append(ReactionClassification(num_reactant, num_product))
        return classifications

    @classmethod 
    def calculateBooleanCompatibilityVector(cls, reference_constraint_nmat:NamedMatrix,
              other_constraint_nmat:NamedMatrix, is_equality:bool=True,
              is_bitwise:bool=False)->Tuple[np.ndarray, NamedMatrix]:  # type: ignore
        """
        Calculates the compatibility of two matrices of constraints.

        Args:
            other: Constraint
            is_equality: bool (default: True) Equality or inequality
            is_bitwise: bool (default: False) Bitwise comparisons

        Returns:
            np.ndarray - vector of booleans
                index n represents i*reference_num_row + jth row in other
            NamedMatrix - matrix of booleans where the rows are the same as the input matrices
        """
        if reference_constraint_nmat.num_column != other_constraint_nmat.num_column:
            raise ValueError("Incompatible number of columns.")
        #
        reference_num_row = reference_constraint_nmat.num_row
        other_num_row = other_constraint_nmat.num_row
        # Check for a null
        num_column = other_constraint_nmat.num_column
        # Calculate the CompatibilityCollection
        #    Create the self array with repeated blocks of each row
        reference_arr = np.concatenate([reference_constraint_nmat.values]*other_num_row, axis=1)
        reference_arr = np.reshape(reference_arr, (reference_num_row*other_num_row, num_column))
        #    Create the other array with all values of each row
        other_arr = np.reshape(other_constraint_nmat.values, other_num_row*num_column)
        other_arr = np.concatenate([other_arr]*reference_num_row)
        other_arr = np.reshape(other_arr, (reference_num_row*other_num_row, num_column))
        # Calculate the compatibility boolean vector
        if is_equality:
            satisfy_arr = reference_arr == other_arr
        else:
            if is_bitwise:
                satisfy_arr = np.bitwise_and(reference_arr, other_arr) == reference_arr
            else:
                satisfy_arr = reference_arr <= other_arr
        satisfy_nmat = NamedMatrix(satisfy_arr,
                           column_names=reference_constraint_nmat.column_names)
        result_arr = np.sum(satisfy_arr, axis=1) == num_column
        return result_arr, satisfy_nmat
    
    def makeCompatibilityCollection(self, target:'Constraint',
          is_diagnostic:bool=False)->CompatibilityCollectionResult:
        """
        Makes a collection of compatible constraints.

        Args:
            target: Constraint
            is_diagnostic: bool (default: False) If True, return a diagnostic matrix

        Returns:
            CompatibilityCollectionResult

        """
        # Initializations
        equality_compatibility_nmat = NULL_NMAT
        inequality_compatibility_numerical_nmat = NULL_NMAT
        inequality_compatibility_bitwise_nmat = NULL_NMAT
        # Calculate the compatibility of the constraints
        if self.equality_nmat != NULL_NMAT:
            is_equality_compatibility = True
            equality_compatibility_arr, equality_compatibility_nmat =  \
                 self.calculateBooleanCompatibilityVector(self.equality_nmat, target.equality_nmat, is_equality=True)
        else:
            is_equality_compatibility = False
        if self.numerical_inequality_nmat != NULL_NMAT:
            is_inequality_compatibility = True
            inequality_compatibility_arr, inequality_compatibility_numerical_nmat = self.calculateBooleanCompatibilityVector(
                  self.numerical_inequality_nmat,
                  target.numerical_inequality_nmat, is_equality=False)
            if self.bitwise_inequality_nmat != NULL_NMAT:
                inequality_bitwise_arr, inequality_compatibility_bitwise_nmat = self.calculateBooleanCompatibilityVector(
                  self.bitwise_inequality_nmat,
                  target.bitwise_inequality_nmat, is_equality=False, is_bitwise=True)
                inequality_compatibility_arr &= inequality_bitwise_arr
        else:
            is_inequality_compatibility = False
        # Calculate the compatibility vector by combining the equality and inequality constraints
        if is_equality_compatibility and is_inequality_compatibility:
            compatibility_arr = equality_compatibility_arr & inequality_compatibility_arr
        elif is_equality_compatibility:
            compatibility_arr = equality_compatibility_arr
        elif is_inequality_compatibility:
            compatibility_arr = inequality_compatibility_arr
        else:
            # No constraints
            compatibility_arr = np.repeat(True, target.num_row*self.num_row)
        # Create the compatibility collection
        compatibility_collection = CompatibilityCollection(self.num_row, target.num_row)
        target_arr = np.array(range(target.num_row))
        for irow in range(self.num_row):
            #  Select the rows in other that are compatible with the row in self
            base_pos = irow*target.num_row
            idxs = np.array(range(base_pos, base_pos+target.num_row))
            sel_idxs = compatibility_arr[idxs]
            target_rows = target_arr[sel_idxs]
            compatibility_collection.add(irow, target_rows)
        if is_diagnostic:
            result = CompatibilityCollectionResult(compatibility_collection=compatibility_collection,
                    equality_compatibility_nmat=equality_compatibility_nmat,
                    inequality_compatibility_numerical_nmat=inequality_compatibility_numerical_nmat,
                    inequality_compatibility_bitwise_nmat=inequality_compatibility_bitwise_nmat,
                    reference_equality_nmat=self.equality_nmat,
                    target_equality_nmat=target.equality_nmat,
                    reference_numerical_inequality_nmat=self.numerical_inequality_nmat,
                    target_numerical_inequality_nmat=target.numerical_inequality_nmat,
                    reference_bitwise_inequality_nmat=self.bitwise_inequality_nmat,
                    target_bitwise_inequality_nmat=target.bitwise_inequality_nmat)
        else:
            result = CompatibilityCollectionResult(compatibility_collection=compatibility_collection)
        return result
    
    def makeSuccessorPredecessorConstraintMatrix(self)->NamedMatrix:
        """Make a marix of count of successor and predecessor counts. Uses
        subclass.one_step_arr to calculate the counts.

        Returns:
            NamedMatrix: Rows are reactions, columns are constraints by count of reaction type.
              <ReactionType>
        """
        NUM_TRAVERSAL = 2
        def makeTraversalCounts(one_step_arr):
            # one_step_arr: N X N where row is current and column is next
            # Returns: N X NUM_TRAVERSAL where row is current and column is count of next
            multistep_arr = one_step_arr.copy()
            # Process the steps
            count_arrs:list = []  # type: ignore
            count_arrs.append(multistep_arr.sum(axis=1).astype(int))
            for _ in range(2, NUM_TRAVERSAL+1):
                multistep_arr = np.sign(np.matmul(multistep_arr, one_step_arr)).astype(int)
                count_arrs.append(multistep_arr.sum(axis=1))
            count_arr = np.array(count_arrs).T
            return count_arr.astype(int)
        #####
        column_names = [f"s_{n+1}" for n in range(NUM_TRAVERSAL)]
        column_names.extend([f"p_{n+1}" for n in range(NUM_TRAVERSAL)])
        # Calculate successor and predecessor arrays
        successor_count_arr = makeTraversalCounts(self.one_step_nmat.values)
        predecessor_count_arr = makeTraversalCounts(self.one_step_nmat.values.T)
        count_arr = np.concatenate([successor_count_arr, predecessor_count_arr], axis=1).astype(int)
        # Make the NamedMatrix
        named_matrix = NamedMatrix(count_arr, row_names=self.one_step_nmat.row_names,
                           row_description=self.one_step_nmat.row_description,
                           column_description="constraints",
                           column_names=column_names)
        return named_matrix
    
    def makeNStepConstraintMatrix(self, num_step:int=1)->NamedMatrix:
        """Make constraints for a one step navigation of the bipartite graph. For species, the successor constraint
        is the count of reaction types reachable from the species as a reactant via a product of a reaction. For
        reactions, the successor constraint is the count of reaction types reachable from the reactions that produce
        Converseley, the predecessor constraint is the count of reaction types that reachable from previous
        species/reaction.

        Args:
            num_step (int): Number of steps to consider.

        Returns:
            NamedMatrix: Rows are species, columns are constraints
        """
        #####
        def makeStepArr(one_step_arr:np.ndarray)->np.ndarray:
            arrays = []
            for idx in range(num_step):
                matrix_power = np.sign(np.linalg.matrix_power(one_step_arr, idx+1))
                #reactant_to_N_product_arr = np.matmul(matrix_power, self.reactant_nmat.values)
                reactant_to_N_product_arr = np.matmul(matrix_power, self.to_reaction_arr)
                forward_classification_arr = np.matmul(reactant_to_N_product_arr, reaction_to_classification_nmat.values)
                arrays.append(forward_classification_arr)
            return np.concatenate(arrays, axis=1)
        #####
        reaction_to_classification_nmat = ReactionClassification.makeReactionClassificationMatrix(
              self.reactant_nmat.column_names, self.reaction_classification_arr)
        classification_names = reaction_to_classification_nmat.column_names
        reactant_to_product_arr = np.matmul(self.reactant_nmat.values, self.product_nmat.values.T)
        forward_step_arr = makeStepArr(self.one_step_nmat.values)
        backward_step_arr = makeStepArr(self.one_step_nmat.values.T)
        # Column names
        successor_column_names = [f"s{n+1}_{c}" for n in range(num_step) for c in classification_names]
        predecessor_column_names = [f"p{n+1}_{c}" for n in range(num_step) for c in classification_names]
        column_names = successor_column_names + predecessor_column_names
        array = np.concatenate([forward_step_arr, backward_step_arr], axis=1)
        # Construct a matrix where the columns are reaction classifications and the rows are reactions
        named_matrix = NamedMatrix(array, row_names=self.row_names,
                           row_description=self.description,
                           column_description='constraints',
                           column_names=column_names)
        return named_matrix
    
#####################################
NULL_CONSTRAINT = Constraint(NULL_NMAT, NULL_NMAT)