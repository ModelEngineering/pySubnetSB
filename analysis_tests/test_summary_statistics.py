import analysis.constants as cn  # type: ignore
from analysis.summary_statistics import SummaryStatistics  # type: ignore

import os
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
                 cn.COL_PROCESS_TIME: float, cn.COL_NUM_PERM: int,
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



if __name__ == '__main__':
    unittest.main()