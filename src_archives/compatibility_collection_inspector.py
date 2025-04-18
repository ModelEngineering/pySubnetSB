'''Provides diagnostics for the compatibility collection result.'''


from pySubnetSB.species_constraint import SpeciesConstraint # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint # type: ignore
from pySubnetSB.named_matrix import NamedMatrix, NULL_NMAT # type: ignore

import collections
import numpy as np # type: ignore
import pandas as pd # type: ignore

TARGET = "target"
REFERENCE = "reference"
EQUALITY = "equality"
INEQUALITY_NUMERICAL = "inequality_numerical"
INEQUALITY_BITWISE = "inequality_bitwise"
CONSTRAINT_TYPES = [EQUALITY, INEQUALITY_NUMERICAL, INEQUALITY_BITWISE]


ColumnDescriptor = collections.namedtuple("ColumnDescriptor", ["column_name", "named_matrix", "constraint_type"])


#####################################
class CompatibilityCollectionInspector(object):

    def __init__(self, reference_network, target_network, is_species:bool=True, is_subnet:bool=False):
        """
        Args:
            reference_network (Network)
            target_network (Network)
            is_species (bool, optional): _description_. Defaults to True.
            is_subnet (bool, optional): check for su
        """
        self.reference_network = reference_network
        self.target_network = target_network
        self.is_species = is_species
        self.is_subnet = is_subnet
        #
        if self.is_species:
            constraint_cls = SpeciesConstraint
        else:
            constraint_cls = ReactionConstraint
        self.reference_constraint = constraint_cls(self.reference_network.reactant_nmat,
              self.reference_network.product_nmat, is_subnet=self.is_subnet)
        self.target_constraint = constraint_cls(self.target_network.reactant_nmat,
              self.target_network.product_nmat, is_subnet=self.is_subnet)
        self.compatibility_collection_result = self.reference_constraint.makeCompatibilityCollection(
              self.target_constraint, is_diagnostic=True)
        # 
        self.column_name_dct = self._makeColumnNameDct()
        self._diagnostic_nmat = NULL_NMAT  # Deferred execution to improve testing _labelReferenceTarget
        self._diagnostic_label_nmat = NULL_NMAT  # Deferred execution to improve testing _labelReferenceTarget

    def _makeColumnNameDct(self)->dict:
        """Creates a dictionay column names and their associated NamedMatrix.

        Raises:
            ValueError: ValueError

        Returns: 
            dict:
                key: REFERENCE | TARGET, <column name>
                value: ColumnDescriptor
        """
        dct = {}
        for column_name in self.compatibility_collection_result.reference_equality_nmat.column_names:
            dct[(REFERENCE, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.reference_equality_nmat,
                  constraint_type=EQUALITY)
        for column_name in self.compatibility_collection_result.reference_numerical_inequality_nmat.column_names:
            dct[(REFERENCE, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.reference_numerical_inequality_nmat,
                  constraint_type=INEQUALITY_NUMERICAL)
        for column_name in self.compatibility_collection_result.reference_bitwise_inequality_nmat.column_names:
            dct[(REFERENCE, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.reference_bitwise_inequality_nmat,
                  constraint_type=INEQUALITY_BITWISE)
        #
        for column_name in self.compatibility_collection_result.target_equality_nmat.column_names:
            dct[(TARGET, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.target_equality_nmat,
                  constraint_type=EQUALITY)
        for column_name in self.compatibility_collection_result.target_numerical_inequality_nmat.column_names:
            dct[(TARGET, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.target_numerical_inequality_nmat,
                  constraint_type=INEQUALITY_NUMERICAL)
        for column_name in self.compatibility_collection_result.target_bitwise_inequality_nmat.column_names:
            dct[(TARGET, column_name)] = ColumnDescriptor(
                  column_name=column_name,
                  named_matrix=self.compatibility_collection_result.target_bitwise_inequality_nmat,
                  constraint_type=INEQUALITY_BITWISE)
        return dct
        
    @property
    def diagnostic_nmat(self)->NamedMatrix:
        if self._diagnostic_nmat is NULL_NMAT:
            self._diagnostic_nmat = self._makeDiagnosticMatrix()
        return self._diagnostic_nmat
    
    @property
    def diagnostic_label_nmat(self)->NamedMatrix:
        if self._diagnostic_label_nmat is NULL_NMAT:
            self._diagnostic_label_nmat = self._makeDiagnosticLabelMatrix()
        return self._diagnostic_label_nmat
    
    @property
    def labelled_diagnostic_nmat(self)->NamedMatrix:
        return NamedMatrix.hstack([self.diagnostic_label_nmat, self.diagnostic_nmat])

    def __bool__(self):
        # Returns True if the diagnostic matrix is not NULL_NMAT.
        return self.diagnostic_nmat is not NULL_NMAT
    
    def _makeDiagnosticLabelMatrix(self)->NamedMatrix:
        """
        Construct labels rows in the cross product of reference and target constraint matrices.
        Reference indices run slower than target, as indicated below for a comparison of species names
        with 2 species (S*) in self and 3 in other (A*).

              referencex  target
              S0           A0
              S0           A1
              S0           A2
              S1           A0
              S1           A1
              S1           A2

        Args:
            target: Constraint
            is_species:boolean (default: True) True if species, False if reactions
        """
        # Get the names of the elements being constrained
        reference_names = self.reference_constraint.row_names
        target_names = self.target_constraint.row_names
        reference_length = len(reference_names)
        target_length = len(target_names)
        # Construct the NamedMatrix
        reference_labels = np.repeat(reference_names, target_length)
        length = reference_labels.shape[0]
        reference_labels = np.reshape(reference_labels, (length, 1))
        target_labels = np.array([target_names]*reference_length).flatten()
        target_labels = np.reshape(target_labels, (length, 1))
        arr = np.concatenate([reference_labels, target_labels], axis=1)
        label_nmat = NamedMatrix(arr, column_names=["reference", "target"])
        return label_nmat
    
    def _makeDiagnosticMatrix(self)->NamedMatrix:
        # Calculates the diagnostic matrix.
        diagnostic_nmat = NULL_NMAT
        for nmat in [self.compatibility_collection_result.equality_compatibility_nmat,
                     self.compatibility_collection_result.inequality_compatibility_numerical_nmat,
              self.compatibility_collection_result.inequality_compatibility_bitwise_nmat]:
            if diagnostic_nmat == NULL_NMAT:
                diagnostic_nmat = nmat
            else:
                if nmat != NULL_NMAT:
                    diagnostic_nmat = NamedMatrix.hstack([diagnostic_nmat, nmat])
        return diagnostic_nmat
    
    def getTrueRowsInDiagnosticMatrix(self)->np.ndarray:
        """Returns the indices of the rows in the diagnostic matrix that are True.

        Returns:
            np.ndarray: Indices of the rows in the diagnostic matrix that are True.
        """
        row_idxs = [n for n in range(self.diagnostic_nmat.num_row) if np.all(self.diagnostic_nmat.values[n, :])]
        return np.array(row_idxs)

    def _getConstraintValue(self, element_name:str, column_name:str, is_reference:bool=True)->float:
        """Returns the values of the constraint

        Args:
            element_name (str): Name of the element (species or reaction).
            column_name (str): Name of the constraint.
            is_reference (bool, optional): True if the reference network. Defaults to True.

        Returns:
            np.ndarray: Column for the NamedMatrix
        """
        if is_reference:
            full_nmat = self.column_name_dct[(REFERENCE, column_name)].named_matrix
        else:
            full_nmat = self.column_name_dct[(TARGET, column_name)].named_matrix
        nmat = full_nmat.getSubNamedMatrix(row_names=[element_name], column_names=[column_name]).named_matrix
        return nmat.values.flatten()[0]
    
    def explainNotCompatible(self, reference_name:str, target_name:str)->pd.DataFrame:
        """Provides information as to why the target_name is not compatible with the reference_name.

        Args:
            reference_name (str)
            target_name (str)

        Returns:
            pd.DataFrame: Indexed by name of constraint
                reference_value: reference value
                target_value: target value
                attrbute_type: result(T/F), eq, ineq_num, ineq_bit
                comparison_result: comparison result (bool)
        """
        REFERENCE_VALUE = "reference_value"
        TARGET_VALUE = "target_value"
        CONSTRAINT_TYPE = "constraint_type"
        COMPARISON_RESULT = "comparison_result"
        # Returns a string representation of the diagnostic matrix.
        reference_target_idxs = [n for n in range(self.diagnostic_label_nmat.num_row)
              if self.diagnostic_label_nmat.values[n, 0] == reference_name and
              self.diagnostic_label_nmat.values[n, 1] == target_name]
        if len(reference_target_idxs) != 1:
            raise ValueError("No diagnostic information for the reference and target names.")
        reference_target_idx = reference_target_idxs[0]
        ser = pd.Series(self.diagnostic_nmat.values[reference_target_idx, :],
                index=self.diagnostic_nmat.column_names)
        false_constraint_column_names = ser.index[ser == False]
        # Construct the DataFrame
        dataframe_column_names = [REFERENCE_VALUE, TARGET_VALUE, CONSTRAINT_TYPE, COMPARISON_RESULT]
        dct:dict = {c: [] for c in dataframe_column_names}
        for false_constraint_column_name in false_constraint_column_names:
            false_constraint_column_idx = self.diagnostic_nmat.getColumnIdx(false_constraint_column_name)
            comparison_result = self.diagnostic_nmat.values[reference_target_idx, false_constraint_column_idx]
            reference_value = self._getConstraintValue(reference_name, false_constraint_column_name,
                  is_reference=True)
            target_value = self._getConstraintValue(target_name, false_constraint_column_name,
                  is_reference=False)
            constraint_type = self.column_name_dct[(REFERENCE, false_constraint_column_name)].constraint_type
            dct[REFERENCE_VALUE].append(reference_value)
            dct[TARGET_VALUE].append(target_value)
            dct[CONSTRAINT_TYPE].append(constraint_type)
            dct[COMPARISON_RESULT].append(comparison_result)
        df = pd.DataFrame(dct, index=false_constraint_column_names)
        return df