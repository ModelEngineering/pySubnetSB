from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore
from pySubnetSB.network import Network  # type: ignore
import pySubnetSB.constants as cn # type: ignore

import numpy as np
import matplotlib.pyplot as plt
import unittest


IGNORE_TEST = True
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
NUM_ITERATION = 1000
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
        self.calculator = SignificanceCalculator(REFERENCE_NETWORK, NUM_TARGET_REACTION,
              NUM_TARGET_SPECIES, identity=IDENTITY)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.calculator.reference_network is not None)
        self.assertEqual(self.calculator.num_target_reaction, NUM_TARGET_REACTION)
        self.assertEqual(self.calculator.num_target_species, NUM_TARGET_SPECIES)
        self.assertEqual(self.calculator.identity, IDENTITY)
    
    def testCalculateSimple(self):
        if IGNORE_TEST:
            return
        result = self.calculator.calculate(NUM_ITERATION, max_num_assignment=MAX_NUM_ASSIGNMENT,
              is_report=IGNORE_TEST)
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
        self.assertTrue(result.frac_induced > 0.1)
    
    def testCalculateComplex(self):
        if IGNORE_TEST:
            return
        reference_network = Network.makeFromAntimonyStr(COMPLEX_MODEL)
        calculator = SignificanceCalculator(reference_network, NUM_TARGET_REACTION,
              NUM_TARGET_SPECIES, identity=IDENTITY)
        result = calculator.calculate(NUM_ITERATION, max_num_assignment=MAX_NUM_ASSIGNMENT,
              is_report=False)
        self.assertTrue(result.frac_induced < 0.1)

    def testPlotSignificance(self):
        # Plots probability of induced network in a random target as the number of iterations increases
        if IGNORE_TEST:
            return
        result = self.calculator.plotSignificance(
              is_report=IGNORE_TEST,
              num_iteration=100, is_plot=False)
        for values in [result.frac_induces, result.frac_truncates]:
            self.assertTrue(len(result.target_sizes), len(values))
        ax = plt.gca()
        ax.set_title("Significance of Induced Subnetwork")
        plt.show()

    def testCalculateOccurrenceProbability(self):
        #if IGNORE_TEST:
        #    return
        reference_network = REFERENCE_NETWORK
        reference_network = Network.makeFromAntimonyStr(COMPLEX_MODEL)
        result = self.calculator.calculateOccurrenceProbability(
              reference_network, num_iteration=1000000, is_report=False,
              max_num_assignment=MAX_NUM_ASSIGNMENT)
        import pdb; pdb.set_trace()
        

if __name__ == '__main__':
    unittest.main()