import analysis.result_accessor as ra # type: ignore
from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn  # type: ignore
import analysis.constants as cn  # type: ignore
import sirn.util as util  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore

import os
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
import tellurium as te  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
OSCILLATOR_DIR = "Oscillators_JUNE_10_B_10507"
SERIALIZED_FILE = "Oscillators_DOE_JUNE_10_17565_serializers.csv"
CLUSTER_RESULT_FILE = "strong100_Oscillators_JUNE_10_B_10507.txt"
ANALYSIS_RESULT_PATH = os.path.join(cn.TEST_DIR, CLUSTER_RESULT_FILE)
MAX_NUM_ASSIGNMENT = 100
CLUSTERED_NETWORK_COLUMN_DCT = {cn.COL_NETWORK_NAME: str, cn.COL_PROCESSING_TIME: float,
      cn.COL_LEN_ASSIGNMENT_COLLECTION: np.int64, cn.COL_IS_INDETERMINATE: np.bool_, cn.COL_COLLECTION_IDX: np.int64}
CLUSTERED_NETWORK_COLLECTION_COLUMN_DCT = {cn.COL_HASH: np.uint64, cn.COL_COLLECTION_IDX: np.int64,
      cn.COL_NUM_NETWORK:np.int64}


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
        self.assertEqual(self.accessor.identity, cnn.ID_STRONG)
        self.assertEqual(self.accessor.max_num_assignment, MAX_NUM_ASSIGNMENT)

    def testDataframe(self):
        if IGNORE_TEST:
            return
        def checkDataFrame(df, column_dct):
            for column, data_type in column_dct.items():
                if not column in df.columns:
                    self.assertTrue(False)
                if not isinstance(df.loc[0, column], data_type):
                    import pdb; pdb.set_trace()
                    self.assertTrue(False)
        #####
        self.assertEqual(len(self.accessor.clustered_network_collection_df.columns),
              len(cn.RESULT_ACCESSOR_CLUSTERED_NETWORK_COLLECTION_COLUMNS))
        self.assertEqual(len(self.accessor.clustered_network_df.columns), len(cn.RESULT_ACCESSOR_CLUSTERED_NETWORK_COLUMNS))
        checkDataFrame(self.accessor.clustered_network_df, CLUSTERED_NETWORK_COLUMN_DCT)
        checkDataFrame(self.accessor.clustered_network_collection_df, CLUSTERED_NETWORK_COLLECTION_COLUMN_DCT)

    def testIterateDir(self):
        if IGNORE_TEST:
            return
        iter = ResultAccessor.iterateDir("sirn_analysis")
        for directory, df in iter:
            self.assertTrue(isinstance(directory, str))
            self.assertTrue(isinstance(df, pd.DataFrame))

    def testIsClusterSubset(self):
        # FIXME: Tests disabled
        return
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
        antimony_str = self.accessor.getAntimonyFromNetworkname("MqCUzoSy_k7iNe0A_1313_9")
        self.assertTrue(isinstance(antimony_str, str))
        rr = te.loada(antimony_str)
        self.assertTrue("RoadRunner" in str(type(rr)))

    def testGetAntimonyFromCollectionidx(self):
        if IGNORE_TEST:
            return
        if not cn.IS_OSCILLATOR_ZIP:
            return
        df = self.accessor.clustered_network_collection_df[
              self.accessor.clustered_network_collection_df[cn.COL_NUM_NETWORK] > 1]
        indices = list(df.index)
        collection_idx = df.loc[indices[0], cn.COL_COLLECTION_IDX]
        antimony_strs = self.accessor.getAntimonyFromCollectionidx(collection_idx)
        self.assertEqual(len(antimony_strs), df.loc[indices[0], cn.COL_NUM_NETWORK])
        for antimony_str in antimony_strs:
            rr = te.loada(antimony_str)
            self.assertTrue("RoadRunner" in str(type(rr)))


    def testGetClusterResultPath(self):
        if IGNORE_TEST:
            return
        path = ResultAccessor.getClusterResultPath(OSCILLATOR_DIR)
        self.assertTrue(os.path.exists(path))

    def testGetNetworkCollectionFromCSVFile(self):
        if IGNORE_TEST:
            return
        file_path = os.path.join(cn.TEST_DIR, "Oscillators_DOE_JUNE_10_17565_serializers.csv")
        network_collection = ResultAccessor.getNetworkCollectionFromCSVFile(file_path)
        self.assertTrue(isinstance(network_collection, NetworkCollection))
        self.assertEqual(len(network_collection.networks), 17565)


if __name__ == '__main__':
    unittest.main(failfast=True)