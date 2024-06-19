import sirn.constants as cn  # type: ignore
from sirn.pmatrix import PMatrix # type: ignore
from sirn.network import Network # type: ignore

import copy
import os
import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
NETWORK_NAME = "test"
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
// Structurally identical with Network1
Ja: S2 -> S2 + S1; k1*S2;
Jb: S1 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK3 = """
// Not structurally identical with Network1 if not is_simple_stoichiometry
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

NETWORK = Network.makeAntimony(NETWORK1, network_name=NETWORK_NAME, is_simple_stoichiometry=False)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = copy.deepcopy(NETWORK)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        for pmatrix in [self.network.reactant_pmatrix, self.network.product_pmatrix]:
            self.assertTrue(isinstance(pmatrix, PMatrix))

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        network = self.network.copy()
        self.assertEqual(self.network, network)

    def testIsStructurallyIdentical1(self):
        # Structurally identical under any definition
        if IGNORE_TEST:
            return
        def test(is_simple_stoichiometry):
            network1 = Network.makeAntimony(NETWORK1, network_name="NETWORK1", is_simple_stoichiometry=is_simple_stoichiometry)
            self.assertTrue(network1.isStructurallyIdentical(network1))
        #
        test(True)
        test(False)

    def testIsStructurallyIdentical2(self):
        # Structurally identical under any definition, a different network
        if IGNORE_TEST:
            return
        def test(is_simple_stoichiometry, test_result):
            network1 = Network.makeAntimony(NETWORK1, network_name="NETWORK1",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            network2 = Network.makeAntimony(NETWORK2, network_name="NETWORK2",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            self.assertEqual(network1.isStructurallyIdentical(network2), test_result)
        #
        test(False, True)
        test(True, True)
    
    def testIsStructurallyIdentical3(self):
        # Structurally identical only for simple stoiometry
        if IGNORE_TEST:
            return
        def test(is_simple_stoichiometry, test_result):
            network1 = Network.makeAntimony(NETWORK1, network_name="NETWORK1",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            network3 = Network.makeAntimony(NETWORK3, network_name="NETWORK3",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            self.assertEqual(network1.isStructurallyIdentical(network3), test_result)
        #
        test(True, True)
        test(False, False)
    
    def testIsStructurallyIdentical4(self):
        # Not structurally identical
        if IGNORE_TEST:
            return
        def test(is_simple_stoichiometry, test_result):
            network1 = Network.makeAntimony(NETWORK1, network_name="NETWORK1",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            network4 = Network.makeAntimony(NETWORK4, network_name="NETWORK4",
                                            is_simple_stoichiometry=is_simple_stoichiometry)
            self.assertEqual(network1.isStructurallyIdentical(network4), test_result)
        #
        test(True, False)
        test(False, False)



if __name__ == '__main__':
    unittest.main()