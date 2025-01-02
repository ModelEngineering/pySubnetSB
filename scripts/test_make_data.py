import pySubnetSB.constants as cn  # type: ignore
import scripts.make_data as md

import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
TEST_SUBNET_BIOMODELS_STRONG_PATH = "/tmp/subnet_biomodels_strong.csv"
TEST_SUBNET_BIOMODELS_STRONG_PRELIMINARY_PATH = "/tmp/subnet_biomodels_strong_preliminary.csv"
TEST_SUBNET_BIOMODELS_WEAK_PATH = "/tmp/subnet_biomodels_weak.csv"
TEST_SUBNET_BIOMODELS_WEAK_PRELIMINARY_PATH = "/tmp/subnet_biomodels_weak_preliminary.csv"
TEST_BIOMODELS_SUMMARY_PATH = "/tmp/biomodels_summary.csv"
TEST_BIOMODELS_SUMMARY_WORKER_PATH = "/tmp/biomodels_summary_worker_0.csv"
REMOVE_FILES:list = [TEST_SUBNET_BIOMODELS_STRONG_PATH, TEST_SUBNET_BIOMODELS_WEAK_PATH,
      TEST_SUBNET_BIOMODELS_STRONG_PRELIMINARY_PATH, TEST_SUBNET_BIOMODELS_WEAK_PRELIMINARY_PATH,
      TEST_BIOMODELS_SUMMARY_PATH, TEST_BIOMODELS_SUMMARY_WORKER_PATH]

#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.remove()

    def tearDown(self):
        self.remove()

    def remove(self):
        for f in REMOVE_FILES:
            if os.path.isfile(f):
                os.remove(f)

    def testMakeSubnetData(self):
        if IGNORE_TEST:
            return
        NUM_ROW = 5
        md.makeSubnetData(cn.FULL_BIOMODELS_STRONG_PATH, TEST_SUBNET_BIOMODELS_STRONG_PATH,
              num_row=NUM_ROW, is_report=IGNORE_TEST)
        self.assertTrue(os.path.isfile(TEST_SUBNET_BIOMODELS_STRONG_PATH))
        df1 = pd.read_csv(TEST_SUBNET_BIOMODELS_STRONG_PATH)
        self.assertEqual(len(df1), NUM_ROW)
        self.assertEqual(np.sum(df1['num_assignment_pair'] > 0), NUM_ROW)
        # Check recovering existing data
        md.makeSubnetData(cn.FULL_BIOMODELS_STRONG_PATH, TEST_SUBNET_BIOMODELS_STRONG_PATH,
              num_row=NUM_ROW, is_report=IGNORE_TEST)
        df2 = pd.read_csv(TEST_SUBNET_BIOMODELS_STRONG_PATH)
        self.assertEqual(len(df2), 2*NUM_ROW)

    def testMakeModelSummary(self):
        if IGNORE_TEST:
            return
        NUM_MODEL = 100
        md.makeModelSummary(cn.BIOMODELS_SERIALIZATION_PATH, TEST_BIOMODELS_SUMMARY_PATH,
            num_worker=1, worker_idx=0, is_report=IGNORE_TEST,
            num_iteration=10, total_model=NUM_MODEL)
        df = pd.read_csv(TEST_BIOMODELS_SUMMARY_WORKER_PATH)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), NUM_MODEL)
        self.assertGreater(np.sum(df[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]), 0)

    """ def testConsolidateBiomodelSummary(self):
        if IGNORE_TEST:
            return
        md.consolidateBiomodelsSummary(cn.BIOMODELS_SUMMARY_MULTIPLE_PATH, TEST_BIOMODELS_SUMMARY_PATH,
              is_report=IGNORE_TEST)
        df = pd.read_csv(TEST_BIOMODELS_SUMMARY_PATH)
        self.assertTrue(isinstance(df, pd.DataFrame)) """
    
    def testCleanSummaryData(self):
        if IGNORE_TEST:
            return
        md.cleanSummaryData(cn.BIOMODELS_SUMMARY_PRELIMINARY_PATH, TEST_BIOMODELS_SUMMARY_PATH,
              is_report=IGNORE_TEST)
        df = pd.read_csv(TEST_BIOMODELS_SUMMARY_PATH)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def testAddEstimatedPOC(self):
        if IGNORE_TEST:
            return
        dff = pd.read_csv(cn.SUBNET_BIOMODELS_STRONG_PRELIMINARY_PATH)
        md.addEstimatedPOC(cn.SUBNET_BIOMODELS_STRONG_PRELIMINARY_PATH, cn.BIOMODELS_SUMMARY_PATH,
              TEST_SUBNET_BIOMODELS_STRONG_PATH, is_report=IGNORE_TEST)
        df = pd.read_csv(TEST_SUBNET_BIOMODELS_STRONG_PATH)
        #plt.scatter(df[cn.D_ESTIMATED_POC_STRONG], df[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG])
        #
#        sel = df['probability_of_occurrence_weak']<0.01
#        dff = df[sel]
#        yvals = [max(v, 1e-5) for v in df['probability_of_occurrence_weak']]
#        plt.hist(-np.log10(yvals), bins=100)
#        plt.show()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), len(dff))


if __name__ == '__main__':
    unittest.main()