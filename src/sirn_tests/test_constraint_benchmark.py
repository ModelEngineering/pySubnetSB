from sirn.constraint_benchmark import ConstraintBenchmark
from sirn.network import Network
from sirn.species_constraint import SpeciesConstraint
from sirn.reaction_constraint import ReactionConstraint

import numpy as np
import re
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_REACTION = 5
NUM_SPECIES = 5
NUM_ITERATION = 10


#############################
# Tests
#############################
class TestBenchmark(unittest.TestCase):

    def setUp(self):
        self.benchmark = ConstraintBenchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION, False)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark.num_reaction, 5)
        self.assertEqual(len(self.benchmark.reference_networks), NUM_ITERATION)
        self.assertEqual(len(self.benchmark.filler_networks), NUM_ITERATION)

    def testGetConstraintClass(self):
        if IGNORE_TEST:
            return
        constraint_class = self.benchmark._getConstraintClass(is_species=True)
        self.assertEqual(constraint_class, SpeciesConstraint)
        constraint_class = self.benchmark._getConstraintClass(is_species=False)
        self.assertEqual(constraint_class, ReactionConstraint)

    def validateBenchmarkDataframe(self, benchmark, df):
        self.assertTrue('time' in df.columns)
        self.assertTrue('num_permutation' in df.columns)
        self.assertEqual(len(df), benchmark.num_iteration)

    def testRun(self):
        if IGNORE_TEST:
            return
        df = self.benchmark.run(is_species=True)
        self.validateBenchmarkDataframe(self.benchmark, df)
        df = self.benchmark.run(is_species=False)
        self.validateBenchmarkDataframe(self.benchmark, df)


if __name__ == '__main__':
    unittest.main()