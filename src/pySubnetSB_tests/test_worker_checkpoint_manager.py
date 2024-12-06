import pySubnetSB.constants as cn  # type: ignore
import pySubnetSB.parallel_subnet_finder_worker as psfw  # type: ignore
from pySubnetSB.network import Network  # type: ignore

import os
import json
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT =  False
SIZE = 10
NUM_NETWORK = 10    
BASE_CHECKPOINT_PATH = os.path.join(cn.TEST_DIR, "test_subnet_finder_checkpoint.csv")
WORKER0_CHECKPOINT_PATH = psfw.WorkerCheckpointManager.makeWorkerCheckpointPath(BASE_CHECKPOINT_PATH, 0)
WORKER1_CHECKPOINT_PATH = psfw.WorkerCheckpointManager.makeWorkerCheckpointPath(BASE_CHECKPOINT_PATH, 1)
REMOVE_FILES:list = [BASE_CHECKPOINT_PATH, WORKER0_CHECKPOINT_PATH, WORKER1_CHECKPOINT_PATH]


#############################
class TestWorkerCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.checkpoint_manager = psfw.WorkerCheckpointManager(BASE_CHECKPOINT_PATH, is_report=IS_PLOT)

    def remove(self):
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)

    def tearDown(self):
        self.remove()

    @staticmethod
    def makeDataframe(num_network:int)->pd.DataFrame:
        # Creates a dataframe used by the WorkerCheckpointManager
        reference_networks = [Network.makeRandomNetworkByReactionType(3, is_prune_species=True) for _ in range(num_network)]
        target_networks = [Network.makeRandomNetworkByReactionType(3, is_prune_species=True) for _ in range(num_network)]
        dct = {cn.FINDER_REFERENCE_NETWORK: [str(n) for n in range(num_network)],
            cn.FINDER_INDUCED_NETWORK: [str(n) for n in range(num_network, 2*num_network)]}
        df = pd.DataFrame(dct)
        df[cn.FINDER_REFERENCE_NAME] = [str(n) for n in reference_networks]
        df[cn.FINDER_TARGET_NAME] = [str(n) for n in target_networks]
        df[cn.FINDER_NAME_DCT] = [json.dumps(dict(a=n)) for n in range(num_network)]
        df[cn.FINDER_IS_TRUNCATED] = [False for _ in range(num_network)]
        return df

    def testRecover(self):
        if IGNORE_TEST:
            return
        num_network = 10
        df = self.makeDataframe(num_network)
        df.loc[0, cn.FINDER_REFERENCE_NETWORK] = ""
        df.loc[0, cn.FINDER_INDUCED_NETWORK] = ""
        self.checkpoint_manager.checkpoint(df)
        result = self.checkpoint_manager.recover()
        self.assertEqual(len(result.full_df), num_network)
        self.assertEqual(len(result.pruned_df), num_network-1)
        self.assertEqual(len(result.processeds), 1)

    # FIXME: test all fields of prune_result
    def testPrune(self):
        if IGNORE_TEST:
            return
        num_network = 10
        df = self.makeDataframe(num_network)
        df.loc[0, cn.FINDER_REFERENCE_NETWORK] = ""
        df.loc[0, cn.FINDER_INDUCED_NETWORK] = ""
        prune_result = self.checkpoint_manager.prune(df)
        self.assertEqual(len(prune_result.reference_names), 1)
        self.assertEqual(len(prune_result.pruned_df), num_network - 1)


if __name__ == '__main__':
    unittest.main(failfast=False)