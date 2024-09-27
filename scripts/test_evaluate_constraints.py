from scripts.evaluate_constraints import Benchmark
from src.sirn.network import Network

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
        self.benchmark = Benchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION, False)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark.num_reaction, 5)
        self.assertEqual(len(self.benchmark.reference_networks), NUM_ITERATION)
        self.assertEqual(len(self.benchmark.filler_networks), NUM_ITERATION)


if __name__ == '__main__':
    unittest.main()