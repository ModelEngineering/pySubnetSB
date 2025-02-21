from src.pySubnetSB.benchmark import Benchmark, C_LOG10_NUM_PERMUTATION, C_TIME  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
import pySubnetSB.constants as cn # type: ignore

import os
import pandas as pd # type: ignore
import itertools
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_REACTION = 5
NUM_SPECIES = 5
FILL_SIZE = 5
NUM_ITERATION = 10


#############################
# Tests
#############################
class TestBenchmark(unittest.TestCase):

    def setUp(self):
        self.benchmark = Benchmark(NUM_REACTION, fill_size=FILL_SIZE,
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
        benchmark = Benchmark(NUM_REACTION, NUM_SPECIES, NUM_ITERATION,
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
        if IGNORE_TEST:
            return
        ax = self.benchmark.plotHeatmap(range(5, 10, 2), range(10, 30, 3), percentile=50, is_plot=IS_PLOT,
              is_no_constraint=True, num_iteration=10, title="No Constraint", num_digit=0,
              font_size=14)
        self.assertTrue("Axes" in str(type(ax)))

    def testCompareConstraints(self):
        if IGNORE_TEST:
            return
        reference_size = 3
        target_size = 8
        fill_size = target_size - reference_size
        benchmark = Benchmark(reference_size, fill_size=fill_size,
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
        benchmark = Benchmark(reference_size, fill_size=fill_size,
                num_iteration=10)
        benchmark.plotCompareConstraints(is_plot=IS_PLOT, is_subnet=True)

    def testCalculateOccurrence(self):
        if IGNORE_TEST:
            return
        pairs = [(3, 3), (3, 10), (10, 10)]
        df = Benchmark.calculateOccurrence(pairs,
              num_iteration=1000, num_replication=5, is_report=IS_PLOT)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), len(pairs))
        self.assertGreaterEqual(df.loc[0, cn.D_MEAN_PROBABILITY], df.loc[1, cn.D_MEAN_PROBABILITY])

    def testCalculateOccurrenceEdgeCases(self):
        if IGNORE_TEST:
            return
        pairs = [(6, 2), (2, 6), (2, 10), (10, 2)]
        df = Benchmark.calculateOccurrence(pairs,
              num_iteration=10, num_replication=5, is_report=IS_PLOT)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), len(pairs))
    
    def testPlotSpeciesReactionHeatmap(self):
        if IGNORE_TEST:
            return
        sizes = list(itertools.product(range(3, 10), range(3, 10)))
        done = False
        if os.path.isfile("benchmark.csv"):
            df = pd.read_csv("benchmark.csv")
            if len(df) == len(sizes):
                done = True
        if not done:
            df = Benchmark.calculateOccurrence(sizes,
               num_iteration=1000, num_replication=5)
            df.to_csv("benchmark.csv", index=False)
        Benchmark.plotSpeciesReactionHeatmap(df, cn.D_MEAN_PROBABILITY, is_plot=IS_PLOT,
              font_size=14)


if __name__ == '__main__':
    unittest.main(failfast=True)