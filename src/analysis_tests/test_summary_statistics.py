import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.summary_statistics import SummaryStatistics  # type: ignore

import os
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
OSCILLATOR_DIR = "Oscillators_May_28_2024_8898"
STRONG = "strong"
MAX_NUM_PERM = 100
CONDITION_DIR = f"{cn.SIRN_DIR}/{STRONG}{MAX_NUM_PERM}"
FILENAME = f"{STRONG}{MAX_NUM_PERM}_{OSCILLATOR_DIR}.txt"
DATA_PATH = os.path.join(CONDITION_DIR, FILENAME)
IS_STRONG = True
MAX_NUM_PERM = 100
COLUMN_DCT = {cn.COL_HASH: int, cn.COL_NETWORK_NAME: str,
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
        self.assertEqual(self.statistics.series.attrs[cn.COL_OSCILLATOR_DIR], OSCILLATOR_DIR)
        
    def testIterateOverOscillatorDirectories(self):
        if IGNORE_TEST:
            return
        series_results = list(self.statistics.iterateOverOscillatorDirectories(CONDITION_DIR))
        for series in series_results:
            self.assertTrue(isinstance(series, pd.Series))
            self.assertTrue(series.attrs[cn.COL_OSCILLATOR_DIR] in cnn.OSCILLATOR_DIRS)

    def testPlotMetricByCondition(self):
        if IGNORE_TEST:
            return
        for metric in [cn.COL_PROCESSING_TIME_TOTAL, cn.COL_IS_INDETERMINATE_MEAN, cn.COL_NUM_PERM_MAX]:
            if "indeterminate" in metric:
                is_log = False
            else:
                is_log = True
            ax, df = self.statistics.plotMetricByConditions(metric,
                identity_types = [True],
                max_num_perms = [1000, 100000],
                sirn_types = [True, False],
                is_log=is_log,
                ylabel="y label",
                legends=["SIRN1000", "SIRN100000", "Naive1000", "Naive100000"],
                is_plot=IS_PLOT,
                )
            self.assertTrue(isinstance(ax, plt.Axes))
            self.assertTrue(isinstance(df, pd.DataFrame))



if __name__ == '__main__':
    unittest.main()