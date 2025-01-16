from pySubnetSB.constraint_benchmark import ConstraintBenchmark, C_LOG10_NUM_PERMUTATION, C_TIME  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore

import pandas as pd # type: ignore
import unittest


IGNORE_TEST = True
IS_PLOT = True
NUM_REACTION = 5
NUM_SPECIES = 5
FILL_SIZE = 5
NUM_ITERATION = 10


#############################
# Tests
#############################
class TestBenchmark(unittest.TestCase):

    def setUp(self):
        self.benchmark = ConstraintBenchmark(NUM_REACTION, fill_size=FILL_SIZE,
              num_iteration=NUM_ITERATION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark.num_reaction, NUM_REACTION)
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
            for is_subnet in [True, False]:
                df = self.benchmark.run(is_species=is_species, is_subnet=is_subnet)
                self.validateBenchmarkDataframe(self.benchmark, df)

    def testRunIsContainsReferenceFalse(self):
        if IGNORE_TEST:
            return
        benchmark = ConstraintBenchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION,
              is_contains_reference=False)
        for is_species in [True, False]:
            for is_subnet in [True, False]:
                df = benchmark.run(is_species=is_species, is_subnet=is_subnet)
                self.validateBenchmarkDataframe(benchmark, df)

    def testPlotConstraintStudy(self):
        if IGNORE_TEST:
            return
        for size in range(9, 10):
            self.benchmark.plotConstraintStudy(size, size, 10, is_plot=IS_PLOT)

    def testPlotHeatmap(self):
        if IGNORE_TEST:
            return
        ax = self.benchmark.plotHeatmap(range(5, 10, 2), range(10, 30, 3), percentile=50, is_plot=IS_PLOT,
                                        num_iteration=10)
        self.assertTrue("Axes" in str(type(ax)))

    def testPlotHeatmapIscontainsFalse(self):
        if IGNORE_TEST:
            return
        ax = self.benchmark.plotHeatmap(range(5, 10, 2), range(10, 30, 3), percentile=50, is_plot=IS_PLOT,
              num_iteration=10, is_contains_reference=False)
        self.assertTrue("Axes" in str(type(ax)))

    def testPlotHeatmapNoConstraint(self):
        #if IGNORE_TEST:
        #    return
        ax = self.benchmark.plotHeatmap(range(5, 10, 2), range(10, 30, 3), percentile=50, is_plot=IS_PLOT,
              is_no_constraint=True, num_iteration=10, title="No Constraint")
        self.assertTrue("Axes" in str(type(ax)))

    def testCompareConstraints(self):
        if IGNORE_TEST:
            return
        reference_size = 3
        target_size = 8
        fill_size = target_size - reference_size
        benchmark = ConstraintBenchmark(reference_size, fill_size=fill_size,
                num_iteration=NUM_ITERATION)
        for is_subnet in [True, False]:
            result = benchmark.compareConstraints(is_subnet=is_subnet)
            for dimension_result in [result.species_dimension_result, result.reaction_dimension_result]:
                self.assertTrue(isinstance(dimension_result.dataframe, pd.DataFrame))
                self.assertGreater(len(dimension_result.dataframe), 0)
            self.assertEqual(result.reference_size, reference_size)
            self.assertEqual(result.target_size, target_size)

    def testPlotCompareConstraints(self):
        if IGNORE_TEST:
            return
        reference_size = 20
        target_size = 100
        fill_size = target_size - reference_size
        benchmark = ConstraintBenchmark(reference_size, fill_size=fill_size,
                num_iteration=1000)
        benchmark.plotCompareConstraints(is_plot=IS_PLOT, is_subnet=True)


if __name__ == '__main__':
    unittest.main(failfast=True)