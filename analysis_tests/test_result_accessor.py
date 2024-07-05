import analysis.result_accessor as ra # type: ignore
from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn  # type: ignore
import analysis.constants as cn  # type: ignore
import sirn.util as util  # type: ignore

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
class TestResultAccessor(unittest.TestCase):

    def setUp(self):
        self.accessor = ResultAccessor(DATA_PATH)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.accessor.antimony_dir, ANTIMONY_DIR)
        self.assertEqual(self.accessor.is_strong, IS_STRONG)
        self.assertEqual(self.accessor.max_num_perm, MAX_NUM_PERM)

    def testDataframe(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.accessor.df.shape[1], len(cn.RESULT_ACCESSOR_COLUMNS))
        self.assertGreater(self.accessor.df.shape[0], 0)
        for column, data_type in COLUMN_DCT.items():
            value = self.accessor.df.loc[0, column]
            if data_type == int:
                self.assertTrue(util.isInt(value))
            else:
                self.assertTrue(isinstance(value, data_type))

    def testIterateDir(self):
        if IGNORE_TEST:
            return
        iter = ResultAccessor.iterateDir("sirn_analysis")
        for directory, df in iter:
            self.assertTrue(isinstance(directory, str))
            self.assertTrue(isinstance(df, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()