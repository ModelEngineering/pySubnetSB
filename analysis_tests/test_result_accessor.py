import analysis.result_accessor as ra # type: ignore
from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn  # type: ignore
import analysis.constants as cn  # type: ignore
import sirn.util as util  # type: ignore

import os
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
import tellurium as te  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
OSCILLATOR_DIR = "Oscillators_May_28_2024_8898"
STRONG = "strong"
MAX_NUM_PERM = 100
FILENAME = f"{STRONG}{MAX_NUM_PERM}_{OSCILLATOR_DIR}.txt"
ANALYSIS_RESULT_PATH = os.path.join(cn.TEST_DIR, FILENAME)
IS_STRONG = True
MAX_NUM_PERM = 100
COLUMN_DCT = {cn.COL_HASH: int, cn.COL_MODEL_NAME: str,
                 cn.COL_PROCESSING_TIME: float, cn.COL_NUM_PERM: int,
           cn.COL_IS_INDETERMINATE: np.bool_, cn.COL_COLLECTION_IDX: int}


#############################
# Tests
#############################
class TestResultAccessor(unittest.TestCase):

    def setUp(self):
        self.accessor = ResultAccessor(ANALYSIS_RESULT_PATH)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.accessor.oscillator_dir, OSCILLATOR_DIR)
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

    def testIsClusterSubset(self):
        if IGNORE_TEST:
            return
        subset_dir = os.path.join(cn.DATA_DIR, "sirn_analysis", "strong10000")
        superset_dir = os.path.join(cn.DATA_DIR, "sirn_analysis", "weak10000")
        missing_dct = ResultAccessor.isClusterSubset(superset_dir, subset_dir)
        self.assertEqual(len(missing_dct[cn.COL_OSCILLATOR_DIR]), 0)
        #
        missing_dct = ResultAccessor.isClusterSubset(subset_dir, superset_dir)
        self.assertGreater(len(missing_dct[cn.COL_OSCILLATOR_DIR]), 0)

    def testGetAntimonyFromModelname(self):
        if IGNORE_TEST:
            return
        if not cn.IS_OSCILLATOR_ZIP:
            return
        antimony_str = self.accessor.getAntimonyFromModelname("MqCUzoSy_k7iNe0A_1313_9")
        self.assertTrue(isinstance(antimony_str, str))
        rr = te.loada(antimony_str)
        self.assertTrue("RoadRunner" in str(type(rr)))

    def testGetAntimonyFromCollectionidx(self):
        if IGNORE_TEST:
            return
        if not cn.IS_OSCILLATOR_ZIP:
            return
        df = self.accessor.df.loc[0, :]
        antimony1_str = self.accessor.getAntimonyFromCollectionidx(df[cn.COL_COLLECTION_IDX])[0]
        antimony2_str = self.accessor.getAntimonyFromModelname(df[cn.COL_MODEL_NAME])
        self.assertEqual(antimony1_str, antimony2_str)

    def testGetClusterResultPath(self):
        if IGNORE_TEST:
            return
        path = self.accessor.getClusterResultPath(OSCILLATOR_DIR)
        self.assertTrue(os.path.exists(path))
        accessor = ResultAccessor(path)
        self.assertGreater(len(accessor.df), 0)


if __name__ == '__main__':
    unittest.main()