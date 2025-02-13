from pySubnetSB.significance_calculator_core import SignificanceCalculatorCore  # type: ignore
from pySubnetSB.network import Network  # type: ignore
import pySubnetSB.constants as cn # type: ignore

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIMPLE_MODEL = """
S1 -> S2; k1*S1
S2 -> S3; k2*S2

S1 = 1
S2 = 0
S3 = 0
k1 = 0.1
k2 = 0.2
"""
COMPLEX_MODEL = """
S1 -> S2; k1*S1
S2 -> S3; k2*S2
S3 -> S1; k3*S3
S3 -> S2; k4*S3
S3 -> S4; k5*S3

S1 = 1
S2 = 0
S3 = 0
k1 = 0.1
k2 = 0.2
k3 = 0.2
k4 = 0.2
k5 = 0.2
"""
NUM_TARGET_NETWORK = 10
MAX_NUM_ASSIGNMENT = int(1e6)
NUM_TARGET_REACTION = 10
NUM_TARGET_SPECIES = 10
IDENTITY = cn.ID_STRONG
REFERENCE_NETWORK = Network.makeFromAntimonyStr(SIMPLE_MODEL)


#############################
# Tests
#############################
class TestSignificanceCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = SignificanceCalculatorCore(NUM_TARGET_SPECIES,
              NUM_TARGET_REACTION, num_target_network=NUM_TARGET_NETWORK)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.calculator.num_target_reaction, NUM_TARGET_REACTION)
        self.assertLessEqual(self.calculator.num_target_species, NUM_TARGET_SPECIES)
        self.assertGreaterEqual(self.calculator.num_target_network, 
            len(self.calculator.target_networks))
    
    def testCalculateSimple(self):
        if IGNORE_TEST:
            return
        for identity in cn.ID_LST:
            result = self.calculator.calculateSubnet(REFERENCE_NETWORK, max_num_assignment=MAX_NUM_ASSIGNMENT,
                is_report=IGNORE_TEST, identity=identity)
            self.assertTrue(result.num_reference_species > 0)
            self.assertTrue(result.num_reference_reaction > 0)
            self.assertEqual(result.num_target_species, NUM_TARGET_SPECIES)
            self.assertEqual(result.num_target_reaction, NUM_TARGET_REACTION)
            self.assertEqual(result.max_num_assignment, MAX_NUM_ASSIGNMENT)
            self.assertEqual(result.identity, identity)
            self.assertTrue(result.num_induced >= 0)
            self.assertTrue(result.num_truncated >= 0)
            self.assertTrue(result.frac_induced >= 0)
            self.assertTrue(result.frac_truncated >= 0)
    
    def testCalculateComplex(self):
        if IGNORE_TEST:
            return
        reference_network = Network.makeFromAntimonyStr(COMPLEX_MODEL)
        calculator = SignificanceCalculatorCore(NUM_TARGET_SPECIES,
              NUM_TARGET_REACTION, num_target_network=1000*NUM_TARGET_NETWORK)
        result = calculator.calculateSubnet(reference_network, max_num_assignment=MAX_NUM_ASSIGNMENT,
              is_report=IS_PLOT)
        self.assertTrue(result.frac_induced < 0.1)
    
    def testCalculateComplexEquality(self):
        if IGNORE_TEST:
            return
        for identity in cn.ID_LST:
            results = []
            for size in [2, 4]:
                reference_network = Network.makeRandomNetworkByReactionType(num_reaction=size,
                      num_species=size, is_exact=True)
                calculator = SignificanceCalculatorCore(size, size, num_target_network=1000)
                result = calculator.calculateEqual(reference_network,
                      max_num_assignment=MAX_NUM_ASSIGNMENT,
                      is_report=IS_PLOT, identity=identity)
                results.append(result)
            self.assertTrue(results[0].frac_induced >= results[1].frac_induced)
        

if __name__ == '__main__':
    unittest.main()