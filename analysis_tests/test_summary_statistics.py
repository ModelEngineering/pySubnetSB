import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.summary_statistics import SummaryStatistics  # type: ignore

import os
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
import unittest


IGNORE_TEST = True
IS_PLOT = True
ANTIMONY_DIR = "Oscillators_May_28_2024_8898"
STRONG = "strong"
MAX_NUM_PERM = 100
CONDITION_DIR = f"{cn.SIRN_DIR}/{STRONG}{MAX_NUM_PERM}"
FILENAME = f"{STRONG}{MAX_NUM_PERM}_{ANTIMONY_DIR}.txt"
DATA_PATH = os.path.join(CONDITION_DIR, FILENAME)
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
        self.assertEqual(self.statistics.series.attrs[cn.META_ANTIMONY_DIR], ANTIMONY_DIR)
        import pdb; pdb.set_trace()

    def testPlotConditionsByOscillatorDirectory(self):
        if IGNORE_TEST:
            return
        root_dir = os.path.join(cn.DATA_DIR, "sirn_analysis")
        condition_dirs = [os.path.join(root_dir, f"weak{n}") for n in cn.MAX_NUM_PERMS]
        self.statistics.plotConditionByOscillatorDirectory(condition_dirs,
                    [cn.COL_PROCESSING_TIME_MEAN], cn.MAX_NUM_PERMS)
        if IS_PLOT:
            plt.show()
        self.statistics.plotConditionByOscillatorDirectory(condition_dirs,
                    [cn.COL_IS_INDETERMINATE_MEAN], cn.MAX_NUM_PERMS)
        if IS_PLOT:
            plt.show()

    def testPlotConditionMetrics(self):
        if IGNORE_TEST:
            return
        for metric in [cn.COL_PROCESSING_TIME, cn.COL_IS_INDETERMINATE, cn.COL_NUM_PERM]:
            self.statistics.plotConditionMetrics(metric, is_plot=IS_PLOT) 

    def testPlotWithMaxNumPerms(self):
        if IGNORE_TEST:
            return
        SummaryStatistics.plotConditionMetrics(cn.COL_PROCESSING_TIME, is_sirn=False,
                max_num_perms=[100, 1000, 10000, 100000],
                identity_types=["strong"], is_plot=IS_PLOT, ylim=[0, 4],
                title="Strong Identity With Naive Algorithm")
        
    def testIterateOverOscillatorDirectories(self):
        if IGNORE_TEST:
            return
        series_results = list(self.statistics.iterateOverOscillatorDirectories(CONDITION_DIR))
        for series in series_results:
            self.assertTrue(isinstance(series, pd.Series))
            self.assertTrue(series.attrs[cn.COL_OSCILLATOR_DIR] in cnn.OSCILLATOR_DIRS)

        
    def testBug1(self):
        if IGNORE_TEST:
            return
        SummaryStatistics.plotConditionMetrics(cn.M_CLUSTER_SIZE_EQ1, is_sirn=True,
        identity_types=["strong"], max_num_perms=[10000], ylim=[0,1],
        title="Strong Identity with SIRN Algorithm")

    def testPlotMetricByCondition(self):
        #if IGNORE_TEST:
        #    return
        ax, df = self.statistics.plotMetricByConditions(cn.COL_PROCESSING_TIME_TOTAL,
                identity_types = [True],
                max_num_perms = [100, 1000],
                sirn_types = [True, False],
                is_log=True,
                )
        self.assertTrue(isinstance(ax, plt.Axes))
        self.assertTrue(isinstance(df, pd.DataFrame))
        if IS_PLOT:
            plt.show()



if __name__ == '__main__':
    unittest.main()