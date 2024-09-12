from sirn.benchmark_runner import BenchmakrRunner # type: ignore
import sirn.constants as cn  # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIZE = 3
EXPANSION_FACTOR = 2
IDENTITY = cn.ID_WEAK


#############################
# Tests
#############################
class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self):
        self.benchmark_runner = BenchmakrRunner(reference_size=SIZE, expansion_factor=EXPANSION_FACTOR,
              identity=IDENTITY)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark_runner.reference_size, SIZE)
        self.assertEqual(self.benchmark_runner.expansion_factor, EXPANSION_FACTOR)
        self.assertEqual(self.benchmark_runner.identity, IDENTITY)

    def testMakeStructurallySimilarExperiment(self):
        if IGNORE_TEST:
            return
        size = 3
        num_iteration = 10
        for expansion_factor in [1, 2, 3]:
            target_size = size*expansion_factor
            for _ in range(num_iteration):
                benchmark_runner = BenchmakrRunner(reference_size=size, expansion_factor=expansion_factor,
                    identity=IDENTITY)
                experiment = benchmark_runner.makeExperiment()
                self.assertEqual(experiment.reference.num_species, SIZE)
                self.assertEqual(experiment.reference.num_reaction, SIZE)
                self.assertEqual(experiment.target.num_species, target_size)
                self.assertEqual(len(experiment.assignment_pair.species_assignment), target_size)
                self.assertEqual(len(experiment.assignment_pair.reaction_assignment), target_size)
                self.assertEqual(experiment.target.num_reaction, target_size)


if __name__ == '__main__':
    unittest.main()