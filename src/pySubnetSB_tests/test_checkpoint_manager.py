from pySubnetSB.checkpoint_manager import CheckpointManager # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import numpy as np
import os  # type: ignore
import pandas as pd  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
PATH = os.path.join(cn.TEST_DIR, "test_checkpoint.csv")
PATH_PAT = os.path.join(cn.TEST_DIR, "test_checkpoint_%d.csv")
IDENTITY_COLUMN = "identity"
REMOVE_FILES:list = [PATH, PATH_PAT % 0, PATH_PAT % 1]
DCT = {IDENTITY_COLUMN: [1, 2, 3], 'a': [1, 2, 3]}
DCT1 = {IDENTITY_COLUMN: [3, 2, 3, 1], 'a': [0, 1, 2, 3]}

#############################
# Tests
#############################
class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.path = PATH
        self.identity_column = IDENTITY_COLUMN
        self.remove()
        self.checkpoint_manager = CheckpointManager(path=self.path, is_report=IS_PLOT)

    def tearDown(self):
        self.remove()

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.exists(path):
                os.remove(path)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        with open(self.path, 'w') as fd:
            fd.write("identity\n")
        self.assertTrue(os.path.isfile(self.path))

    def testCheckpoint(self):
        if IGNORE_TEST:
            return
        def test(size):
            df = pd.DataFrame(range(size))
            self.checkpoint_manager.checkpoint(df)
            self.assertTrue(os.path.isfile(self.path))
            df = pd.read_csv(self.path)
            self.assertEqual(len(df), size)
        #
        test(3)
        test(6)

    def testRecover(self):
        if IGNORE_TEST:
            return
        df1 = pd.DataFrame(DCT)
        self.checkpoint_manager.checkpoint(df1)
        df2 = self.checkpoint_manager.recover()
        df = pd.concat([df1, df2], ignore_index=True)
        self.checkpoint_manager.checkpoint(df)
        df3 = self.checkpoint_manager.recover()
        self.assertTrue(df.equals(df3))

    def testMergeCheckpoint(self):
        if IGNORE_TEST:
            return
        dfs = [pd.DataFrame(DCT), pd.DataFrame(DCT)]
        lengths = [len(df) for df in dfs]
        checkpoint_managers = [CheckpointManager(path=PATH_PAT % n, is_report=IS_PLOT) for n in range(len(dfs))]
        [cm.checkpoint(df) for cm, df in zip(checkpoint_managers, dfs)]
        #
        self.checkpoint_manager.mergeCheckpoint(checkpoint_managers)
        df = self.checkpoint_manager.recover()
        self.assertEqual(len(df), np.sum(lengths))


if __name__ == '__main__':
    unittest.main()