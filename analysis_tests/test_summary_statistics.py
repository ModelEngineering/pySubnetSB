import analysis.constants as cn  # type: ignore
from analysis.summary_statistics import SummaryStatistics  # type: ignore
import sirn.constants as cnn  # type: ignore

import os
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
ANTIMONY_DIR = "Oscillators_May_28_2024_8898"
STRONG = "strong"
MAX_NUM_PERM = 100
FILENAME = f"{STRONG}{MAX_NUM_PERM}_{ANTIMONY_DIR}.txt"
DATA_PATH = os.path.join(cn.TEST_DIR, FILENAME)
IS_STRONG = True
MAX_NUM_PERM = 100
COLUMN_DCT = {cn.COL_HASH: int, cn.COL_MODEL_NAME: str,
                 cn.COL_PROCESSING_TIME: float, cn.COL_NUM_PERM: int,
           cn.COL_IS_INDETERMINATE: np.bool_, cn.COL_COLLECTION_IDX: int}


#############################
# Tests
#############################
class TestSummaryStatistics(unittest.TestCase):

    def setUp(self):
        self.statistics = SummaryStatistics(DATA_PATH)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.statistics.df.attrs[cn.META_ANTIMONY_DIR], ANTIMONY_DIR)

    def testPlotComparisonBars(self):
        if IGNORE_TEST:
            return
        max_num_perms = [100, 1000, 10000, 100000, 1000000]
        root_dir = os.path.join(cn.DATA_DIR, "sirn_analysis")
        measurement_dirs = [os.path.join(root_dir, f"weak{n}") for n in max_num_perms]
        self.statistics.plotComparisonBars(measurement_dirs, [cn.COL_PROCESSING_TIME],
                                                   max_num_perms)
        if IS_PLOT:
            plt.show()
        self.statistics.plotComparisonBars(measurement_dirs, [cn.COL_IS_INDETERMINATE],
                                           max_num_perms)
        if IS_PLOT:
            plt.show()

    def testPlotMetricComparison(self):
        if IGNORE_TEST:
            return
        for metric in [cn.COL_PROCESSING_TIME, cn.COL_IS_INDETERMINATE, cn.COL_NUM_PERM]:
            self.statistics.plotMetricComparison(metric, is_plot=IS_PLOT) 



if __name__ == '__main__':
    unittest.main()