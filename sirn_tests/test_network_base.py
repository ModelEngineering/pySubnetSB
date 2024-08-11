import sirn.constants as cn  # type: ignore
from sirn.network_base import NetworkBase # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import numpy as np
import copy
import tellurium as te  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
NETWORK_NAME = "test"
BIG_NETWORK = """
J0: S1 -> S1 + S2; k1*S1;
J1: S2 -> S3; k2*S2;
J2: S3 -> ; k3*S3;
J3: S3 -> S4; k4*S3;
J4: S4 -> S5; k5*S4;

k1 = 1
k2 = 1
k3 = 1
k4 = 1
k5 = 1
S1 = 10
S2 = 0
S3 = 0
S4 = 0
S5 = 0
"""
NETWORK1 = """
J0: S1 -> S1 + S2; k1*S1;
J1: S2 -> S3; k2*S2;

k1 = 1
k2 = 1
S1 = 10
S2 = 0
S3 = 0
"""
NETWORK2 = """
// Strong structural identity with Network1
Ja: S2 -> S2 + S1; k1*S2;
Jb: S1 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK3 = """
// Weak sructural identity with Network1
Ja: 2 S1 -> 2 S1 + S2; k1*S1*S1
Jb: S2 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK4 = """
// Not structurally identical with Network1
Ja: S1 -> S2; k1*S2;
Jb: S1 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = copy.deepcopy(NETWORK)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        for mat in [self.network.reactant_mat, self.network.product_mat]:
            self.assertTrue(isinstance(mat, NamedMatrix))
        self.assertTrue("int" in str(type(self.network.weak_hash)))
        self.assertTrue("int" in str(type(self.network.strong_hash)))

    def testGetNetworkMatrix(self):
        if IGNORE_TEST:
            return
        reactant_mat = self.network.getNetworkMatrix(
               matrix_type=cn.MT_STANDARD,
               orientation=cn.OR_SPECIES,
               participant=cn.PR_REACTANT,
                identity=cn.ID_STRONG)
        self.assertTrue(isinstance(reactant_mat, NamedMatrix))
        df = reactant_mat.dataframe
        self.assertTrue(df.columns.name == "reactions")
        self.assertTrue(df.index.name == "species")
    
    def testGetNetworkMatrixMultiple(self):
        if IGNORE_TEST:
            return
        count = 0
        for i_matrix_type in cn.MT_LST:
                for i_orientation in cn.OR_LST:
                    for i_identity in cn.ID_LST:
                        for i_participant in cn.PR_LST:
                            if i_identity == cn.ID_WEAK:
                                if i_participant == cn.PR_PRODUCT:
                                    continue
                                else:
                                    i_participant = None
                            mat = self.network.getNetworkMatrix(i_matrix_type, i_orientation, i_participant, i_identity)
                            count += 1
                            if i_matrix_type == cn.MT_STANDARD:
                                self.assertTrue(isinstance(mat, NamedMatrix))
                                if i_orientation == cn.OR_SPECIES:
                                    self.assertTrue(mat.dataframe.index.name == "species")
                                else:
                                    self.assertTrue(mat.dataframe.index.name == "reactions")
                            elif i_matrix_type == cn.MT_SINGLE_CRITERIA:
                                self.assertTrue(isinstance(mat, SingleCriteriaCountMatrix))
                            elif i_matrix_type == cn.MT_PAIR_CRITERIA:
                                self.assertTrue(isinstance(mat, PairCriteriaCountMatrix))
                            else:
                                self.assertTrue(False)
        self.assertEqual(count, 18)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        network = self.network.copy()
        self.assertEqual(self.network, network)
        
    def testRandomize(self):
        return
        if IGNORE_TEST:
            return
        def test(structural_identity_type):
            for idx in range(10):
                big_network = NetworkBase.makeFromAntimonyStr(BIG_NETWORK)
                network = big_network.randomize(structural_identity_type=structural_identity_type)
                if structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_STRONG:
                    result = big_network.isStructurallyIdentical(network,
                                        is_structural_identity_weak=False)
                    self.assertTrue(result.is_structural_identity_strong)
                elif structural_identity_type == cn.STRUCTURAL_IDENTITY_TYPE_WEAK:
                    result = big_network.isStructurallyIdentical(network,
                                        is_structural_identity_weak=True)
                    self.assertTrue(result.is_structural_identity_weak)
                else:  # cn.STRUCTURAL_IDENTITY_TYPE_NOT
                    result = big_network.isStructurallyIdentical(network,
                                        is_structural_identity_weak=True)
                    self.assertFalse(result.is_excessive_perm)
        #
        test(cn.STRUCTURAL_IDENTITY_TYPE_STRONG)
        test(cn.STRUCTURAL_IDENTITY_TYPE_WEAK)
        test(cn.STRUCTURAL_IDENTITY_TYPE_NOT)

if __name__ == '__main__':
    unittest.main(failfast=True)