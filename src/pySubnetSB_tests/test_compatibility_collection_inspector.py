from pySubnetSB.constraint import Constraint, ReactionClassification, CompatibilityCollection    # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB.compatibility_collection_inspector import CompatibilityCollectionInspector  # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_SPECIES, NUM_REACTION = 3, 3
SPECIES_NAMES = ["S" + str(i) for i in range(NUM_SPECIES)]
REACTION_NAMES = ["J" + str(i) for i in range(NUM_REACTION)]
REFERENCE_NETWORK = Network.makeRandomNetworkByReactionType(NUM_SPECIES, NUM_REACTION)
TARGET_NETWORK = REFERENCE_NETWORK.fill(num_fill_reaction=1, num_fill_species=1)


#############################
class TestCompatibilityCollectionInspector(unittest.TestCase):

    def setUp(self):
        self.inspector = CompatibilityCollectionInspector(REFERENCE_NETWORK, TARGET_NETWORK, is_species=True)

    def testMakeDiagnosticLabelMatrix(self):
        if IGNORE_TEST:
            return
        label_nmat = self.inspector._makeDiagnosticLabelMatrix()
        self.assertEqual(len(label_nmat), REFERENCE_NETWORK.num_species * TARGET_NETWORK.num_species)
        self.assertEqual(label_nmat.num_column, 2)
    
    def testGetTrueRowsInDiagnosticMatrix(self):
        if IGNORE_TEST:
            return
        fill_size = 3
        for is_species in [True, False]:
            for size in range(4, 10):
                reference_network = Network.makeRandomNetworkByReactionType(size, size)
                target_network = reference_network.fill(num_fill_reaction=fill_size, num_fill_species=fill_size)
                inspector = CompatibilityCollectionInspector(reference_network, target_network, is_species=is_species)
                true_idxs = inspector.getTrueRowsInDiagnosticMatrix()
                pair_arr = inspector.diagnostic_label_nmat.values[true_idxs, :]
                num_match = np.sum([p[0] == p[1] for p in pair_arr])
                self.assertGreaterEqual(num_match, reference_network.num_species)

    def testCalculateDiagnosticMatrix(self):
        if IGNORE_TEST:
            return
        diag_nmat = self.inspector._makeDiagnosticMatrix()
        self.assertEqual(diag_nmat.num_row, REFERENCE_NETWORK.num_species * TARGET_NETWORK.num_species)

    def testMakeColumnNameDct(self):
        if IGNORE_TEST:
            return
        column_name_dct = self.inspector._makeColumnNameDct()
        keys = list(column_name_dct.keys())
        value = column_name_dct[keys[0]]
        self.assertTrue(isinstance(column_name_dct, dict))
        self.assertTrue(isinstance(value.column_name, str))
        self.assertTrue(isinstance(value.named_matrix, NamedMatrix))
        self.assertTrue(isinstance(value.constraint_type, str))

    def testGetConstraintValue(self):
        if IGNORE_TEST:
            return
        element_name = "S0"
        for key, descriptor in self.inspector.column_name_dct.items():
            column_name = key[1]
            is_reference = key[0] == "reference"
            nmat = descriptor.named_matrix
            row_idx = list(nmat.row_names).index(element_name)
            column_idx = list(nmat.column_names).index(column_name)
            value = self.inspector._getConstraintValue(element_name, column_name, is_reference)
            self.assertTrue(np.all(nmat.values[row_idx, column_idx] == value))


if __name__ == '__main__':
    unittest.main(failfast=True)