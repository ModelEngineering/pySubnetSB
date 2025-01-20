from pySubnetSB.identity_hash_benchmark import IdentityHashBenchmark, COLUMNS  # type: ignore

import pandas as pd # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
REACTION_NUMS  = [2, 5, 10, 15]
SPECIES_NUMS = [2, 5, 10, 15]
NUM_NETWORK = 100


#############################
# Tests
#############################
class TestBenchmark(unittest.TestCase):

    def setUp(self):
        self.benchmark = IdentityHashBenchmark(SPECIES_NUMS, REACTION_NUMS, num_network=NUM_NETWORK)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark.num_network, NUM_NETWORK)

    def testCalculateHashStatistics(self):
        if IGNORE_TEST:
            return
        df = self.benchmark.calculateHashStatistics()
        trues = [c in df.columns for c in COLUMNS]
        self.assertTrue(all(trues))
        self.assertEqual(len(df), len(SPECIES_NUMS) * len(REACTION_NUMS))

    def testPlotHashStatistics(self):
        if IGNORE_TEST:
            return
        _ = self.benchmark.plotHashStatistics(font_size=14, is_plot=IS_PLOT)

    def testPlotHashStatisticsScale(self):
        if IGNORE_TEST:
            return
        species_nums = range(2, 22, 2)
        reaction_nums = range(2, 22, 2)
        benchmark = IdentityHashBenchmark(species_nums, reaction_nums, num_network=1000)
        _ = benchmark.plotHashStatistics(font_size=14, is_plot=IS_PLOT)



if __name__ == '__main__':
    unittest.main(failfast=True)