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
NETWORK = Network.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)


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

    def evaluateIsStructurallyIdentical(self, network1, network2, is_simple_stoichiometry, expected_result):
        network1 = Network.makeFromAntimonyStr(network1, network_name="NETWORK1")
        network2 = Network.makeFromAntimonyStr(network2, network_name="NETWORK2")
        self.assertEqual(network1.isStructurallyIdentical(network2,
             is_simple_stoichiometry=is_simple_stoichiometry), expected_result)
        
    def testRandomize(self):
        if IGNORE_TEST:
            return
        def test(is_structurally_identical):
            for idx in range(10):
                big_network = Network.makeFromAntimonyStr(BIG_NETWORK)
                network = big_network.randomize(is_structurally_identical=is_structurally_identical)
                if is_structurally_identical:
                    if not big_network.isStructurallyIdentical(network):
                        import pdb; pdb.set_trace( )
                    self.assertTrue(big_network.isStructurallyIdentical(network))
                else:
                    if big_network.isStructurallyIdentical(network):
                        import pdb; pdb.set_trace( )
                    self.assertFalse(big_network.isStructurallyIdentical(network))
        #
        test(True)
        test(False)

    def testIsStructurallyIdentical(self):
        # Structurally identical under any definition
        if IGNORE_TEST:
            return
        def test(is_simple_stoichiometry):
            network1 = Network.makeFromAntimonyFile(NETWORK1, network_name="NETWORK1", is_simple_stoichiometry=is_simple_stoichiometry)
            self.assertTrue(network1.isStructurallyIdentical(network1))
        #
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK1, True, True)
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK1, False, True)

    def testIsStructurallyIdentical2(self):
        # Structurally identical under any definition, a different network
        if IGNORE_TEST:
            return
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK2, True, True)
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK2, False, True)
    
    def testIsStructurallyIdentical3(self):
        # Structurally identical only for simple stoiometry
        if IGNORE_TEST:
            return
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK3, True, True)
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK3, False, False)
    
    def testIsStructurallyIdentical4(self):
        # Not structurally identical
        if IGNORE_TEST:
            return
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK4, True, False)
        self.evaluateIsStructurallyIdentical(NETWORK1, NETWORK4, False, False)



if __name__ == '__main__':
    unittest.main()