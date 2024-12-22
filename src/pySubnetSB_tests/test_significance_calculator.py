from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore
from pySubnetSB.network import Network  # type: ignore
import pySubnetSB.constants as cn # type: ignore

import numpy as np
import unittest


IGNORE_TEST = True
IS_PLOT = False
MODEL = """
S1 -> S2; k1*S1
S2 -> S3; k2*S2

S1 = 1
S2 = 0
S3 = 0
k1 = 0.1
k2 = 0.2
"""
NUM_ITERATION = 1000
MAX_NUM_ASSIGNMENT = 1000
NUM_TARGET_REACTION = 10
NUM_TARGET_SPECIES = 10
IDENTITY = cn.ID_STRONG
REFERENCE_NETWORK = Network.makeFromAntimonyStr(MODEL)


#############################
# Tests
#############################
class TestSignificanceCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = SignificanceCalculator(REFERENCE_NETWORK, NUM_TARGET_REACTION,
              NUM_TARGET_SPECIES, identity=IDENTITY)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.calculator.reference_network is not None)
        self.assertEqual(self.calculator.num_target_reaction, NUM_TARGET_REACTION)
        self.assertEqual(self.calculator.num_target_species, NUM_TARGET_SPECIES)
        self.assertEqual(self.calculator.identity, IDENTITY)
    
    def testCalculate(self):
        #if IGNORE_TEST:
        #    return
        result = self.calculator.calculate(NUM_ITERATION, max_num_assignment=MAX_NUM_ASSIGNMENT)
        self.assertTrue(result.num_reference_species > 0)
        self.assertTrue(result.num_reference_reaction > 0)
        self.assertEqual(result.num_target_species, NUM_TARGET_SPECIES)
        self.assertEqual(result.num_target_reaction, NUM_TARGET_REACTION)
        self.assertEqual(result.num_iteration, NUM_ITERATION)
        self.assertEqual(result.max_num_assignment, MAX_NUM_ASSIGNMENT)
        self.assertEqual(result.identity, IDENTITY)
        self.assertTrue(result.num_induced >= 0)
        self.assertTrue(result.num_truncated >= 0)
        self.assertTrue(result.frac_induced >= 0)
        self.assertTrue(result.frac_truncated >= 0)
        print(result.frac_induced, result.frac_truncated)
        

if __name__ == '__main__':
    unittest.main()