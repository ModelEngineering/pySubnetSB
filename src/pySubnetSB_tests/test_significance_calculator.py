from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore
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
NUM_ITERATION = 10
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
        self.calculator = SignificanceCalculator(REFERENCE_NETWORK, NUM_TARGET_SPECIES,
              NUM_TARGET_REACTION, identity=IDENTITY)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.calculator.reference_network is not None)
        self.assertLessEqual(self.calculator.num_target_reaction, NUM_TARGET_REACTION)
        self.assertLessEqual(self.calculator.num_target_species, NUM_TARGET_SPECIES)
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
        self.assertEqual(result.max_num_assignment, MAX_NUM_ASSIGNMENT)
        self.assertEqual(result.identity, IDENTITY)
        self.assertTrue(result.num_induced >= 0)
        self.assertTrue(result.num_truncated >= 0)
        self.assertTrue(result.frac_induced >= 0)
        self.assertTrue(result.frac_truncated >= 0)
    
    def testCalculateComplex(self):
        if IGNORE_TEST:
            return
        reference_network = Network.makeFromAntimonyStr(COMPLEX_MODEL)
        calculator = SignificanceCalculator(reference_network, NUM_TARGET_SPECIES,
              NUM_TARGET_REACTION, identity=IDENTITY)
        result = calculator.calculate(NUM_ITERATION, max_num_assignment=MAX_NUM_ASSIGNMENT,
              is_report=False)
        self.assertTrue(result.frac_induced < 0.1)

    def testPlotSignificance(self):
        # Plots probability of induced network in a random target as the number of iterations increases
        if IGNORE_TEST:
            return
        result = self.calculator.plotSignificance(
              is_report=IGNORE_TEST,
              num_iteration=5, is_plot=False)
        for values in [result.frac_induces, result.frac_truncates]:
            self.assertTrue(len(result.target_sizes), len(values))

    def testCalculateOccurrenceProbability(self):
        if IGNORE_TEST:
            return
        reference_network = REFERENCE_NETWORK
        reference_network = Network.makeFromAntimonyStr(COMPLEX_MODEL)
        result = self.calculator.calculateNetworkOccurrenceProbability(
              reference_network, num_iteration=NUM_ITERATION, is_report=False,
              max_num_assignment=MAX_NUM_ASSIGNMENT)
        self.assertTrue(result[0] < 0.01)

    def testPlotProbabilityOfOccurrence(self):
        if IGNORE_TEST:
            return
        size = 100
        num_reactions = np.random.randint(1, 10, size=size)
        num_species = np.random.randint(1, 10, size=size)
        values = np.random.uniform(0, 1, size=size)
        df = pd.DataFrame({'num_reaction': num_reactions, 'num_species': num_species,
                           'value': values})
        self.calculator.plotProbabilityOfOccurrence(df, column= 'value', is_plot=IS_PLOT)
        

if __name__ == '__main__':
    unittest.main()