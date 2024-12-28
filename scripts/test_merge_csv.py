import pySubnetSB.constants as cn  # type: ignore
from scripts.merge_csv import merge

import time
import os
import pandas as pd # type: ignore
import numpy as np
import unittest
from functools import cmp_to_key


IGNORE_TEST = False
IS_PLOT = False
NUM_CSV = 3
FILE_STR = __file__.split("/")[-1].split(".")[0]
TMP_DIR = "/tmp"
CSV_PATHS = [os.path.join(cn.SCRIPT_DIR, f"{FILE_STR}_data_{i}.csv") for i in range(NUM_CSV)]
OUTPUT_PATH = os.path.join(TMP_DIR, "test_merge_csv_output.csv")
TEST_SUBNET_BIOMODELS_STRONG_PATH = "/tmp/subnet_biomodels_strong.csv"
TEST_SUBNET_BIOMODELS_WEAK_PATH = "/tmp/subnet_biomodels_weak.csv"
TEST_BIOMODELS_SUMMARY_PATH = "/tmp/biomodels_summary.csv"
REMOVE_FILES = [OUTPUT_PATH, TEST_SUBNET_BIOMODELS_STRONG_PATH,
      TEST_SUBNET_BIOMODELS_WEAK_PATH, TEST_BIOMODELS_SUMMARY_PATH]

#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.makeCSVFiles()

    def tearDown(self):
        self.remove()

    def remove(self):
        for f in REMOVE_FILES:
            if os.path.isfile(f):
                os.remove(f)

    def makeCSVFiles(self):
        for csv_path in CSV_PATHS:
            df = pd.DataFrame(np.random.randint(0, 15, (5, 3)))
            df.to_csv(csv_path, index=False)

    def testMerge(self):
        if IGNORE_TEST:
            return
        merge(directory=cn.SCRIPT_DIR, file_str=FILE_STR, output_path=OUTPUT_PATH)
        df1 = pd.concat([pd.read_csv(f) for f in CSV_PATHS]).reset_index(drop=True)
        df2 = pd.read_csv(OUTPUT_PATH)
        for column in df1.columns:
            diff = set(df1[column]).symmetric_difference(df2[column])
            self.assertEqual(len(diff), 0)
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    unittest.main()