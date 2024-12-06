from pySubnetSB.constraint_benchmark import ConstraintBenchmark, C_LOG10_NUM_PERMUTATION, C_TIME  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore

import pandas as pd # type: ignore
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
        self.benchmark = ConstraintBenchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark.num_reaction, 5)
        self.assertEqual(len(self.benchmark.reference_networks), NUM_ITERATION)
        self.assertEqual(len(self.benchmark.target_networks), NUM_ITERATION)

    def testGetConstraintClass(self):
        if IGNORE_TEST:
            return
        constraint_class = self.benchmark._getConstraintClass(is_species=True)
        self.assertEqual(constraint_class, SpeciesConstraint)
        constraint_class = self.benchmark._getConstraintClass(is_species=False)
        self.assertEqual(constraint_class, ReactionConstraint)

    def validateBenchmarkDataframe(self, benchmark, df):
        self.assertTrue(C_TIME in df.columns)
        self.assertTrue(C_LOG10_NUM_PERMUTATION in df.columns)
        self.assertEqual(len(df), benchmark.num_iteration)

    def testRun(self):
        if IGNORE_TEST:
            return
        for is_species in [True, False]:
            for is_subset in [True, False]:
                df = self.benchmark.run(is_species=is_species, is_subset=is_subset)
                self.validateBenchmarkDataframe(self.benchmark, df)

    def testRunIsContainsReferenceFalse(self):
        if IGNORE_TEST:
            return
        benchmark = ConstraintBenchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION,
              is_contains_reference=False)
        for is_species in [True, False]:
            for is_subset in [True, False]:
                df = benchmark.run(is_species=is_species, is_subset=is_subset)
                self.validateBenchmarkDataframe(benchmark, df)

    def testPlotConstraintStudy(self):
        if IGNORE_TEST:
            return
        for size in range(9, 10):
            self.benchmark.plotConstraintStudy(size, size, 10, is_plot=IS_PLOT)

    def testPlotHeatmap(self):
        if IGNORE_TEST:
            return
        df = self.benchmark.plotHeatmap(range(5, 15, 5), range(10, 30, 10), percentile=50, is_plot=IS_PLOT,
                                        num_iteration=300)
        self.assertTrue(isinstance(df, pd.DataFrame))


if __name__ == '__main__':
    unittest.main(failfast=True)