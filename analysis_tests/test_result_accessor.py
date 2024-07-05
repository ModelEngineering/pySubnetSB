import analysis.result_accessor as ra # type: ignore
from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn  # type: ignore
import analysis.constants as cn  # type: ignore
import sirn.util as util  # type: ignore

import os
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
ra.COLUMN_DCT = {ra.COL_HASH: int, ra.COL_MODEL_NAME: str,
                 ra.COL_PROCESS_TIME: float, ra.COL_NUM_PERM: int,
           ra.COL_IS_INDETERMINATE: np.bool_, ra.COL_COLLECTION_IDX: int}


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
        self.assertEqual(self.accessor.df.shape[1], len(ra.COLUMNS))
        self.assertGreater(self.accessor.df.shape[0], 0)
        for column, data_type in ra.COLUMN_DCT.items():
            value = self.accessor.df.loc[0, column]
            if data_type == int:
                self.assertTrue(util.isInt(value))
            else:
                self.assertTrue(isinstance(value, data_type))


if __name__ == '__main__':
    unittest.main()