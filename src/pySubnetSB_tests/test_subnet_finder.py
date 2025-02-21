import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.subnet_finder import SubnetFinder  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.worker_checkpoint_manager import WorkerCheckpointManager # type: ignore

import json
import os
import pandas as pd # type: ignore
import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT =  False
SIZE = 3
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")
CHECKPOINT_PATH = os.path.join(cn.DATA_DIR, "test_subnet_finder_checkpoint.csv")


#############################
# Tests
#############################
def makeDataframe(num_network:int)->pd.DataFrame:
    # Creates a dataframe used by the CheckpointManager
    reference_networks = [Network.makeRandomNetworkByReactionType(3, is_prune_species=True) for _ in range(num_network)]
    target_networks = [Network.makeRandomNetworkByReactionType(3, is_prune_species=True) for _ in range(num_network)]
    dct = {cn.FINDER_REFERENCE_NETWORK: [str(n) for n in range(num_network)],
           cn.FINDER_INDUCED_NETWORK: [str(n) for n in range(num_network, 2*num_network)]}
    df = pd.DataFrame(dct)
    df[cn.FINDER_REFERENCE_NAME] = [str(n) for n in reference_networks]
    df[cn.FINDER_TARGET_NAME] = [str(n) for n in target_networks]
    df[cn.FINDER_NAME_DCT] = [json.dumps(dict(a=n)) for n in range(num_network)]
    return df


#############################
class TestSubnetFinder(unittest.TestCase):

    def setUp(self):
        self.reference = Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
        self.target = self.reference.fill(num_fill_reaction=SIZE, num_fill_species=SIZE)
        self.finder = SubnetFinder.makeFromCombinations(reference_networks=[self.reference], target_networks=[self.target],
              identity=cn.ID_WEAK)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.finder.network_pairs), 0)

    def testFindSimple(self):
        if IGNORE_TEST:
            return
        df = self.finder.find(is_report=IS_PLOT)
        self.assertEqual(len(df), 1)

    def testFindScale(self):
        if IGNORE_TEST:
            return
        NUM_REFERENCE_MODEL = 10
        NUM_EXTRA_TARGET_MODEL = 10
        NETWORK_SIZE = 10
        fill_size = 3
        # Construct the models
        reference_models = [Network.makeRandomNetworkByReactionType(NETWORK_SIZE, is_prune_species=True)
              for _ in range(NUM_REFERENCE_MODEL)]
        target_models = [r.fill(num_fill_reaction=fill_size, num_fill_species=fill_size) for r in reference_models]
        # Add extra target models
        target_models += [Network.makeRandomNetworkByReactionType(NETWORK_SIZE, is_prune_species=True)
              for _ in range(NUM_EXTRA_TARGET_MODEL)]
        # Do the search
        finder = SubnetFinder.makeFromCombinations(reference_networks=reference_models, target_networks=target_models,
              identity=cn.ID_STRONG)
        df = finder.find(is_report=IS_PLOT)
        prune_df = WorkerCheckpointManager.prune(df).pruned_df
        self.assertEqual(len(prune_df), NUM_REFERENCE_MODEL)
    
    def testFindFromDirectories(self):
        if IGNORE_TEST:
            return
        df = SubnetFinder.findFromDirectories(MODEL_DIR, MODEL_DIR, identity=cn.ID_WEAK, is_report=IS_PLOT)
        num_unique = len(df[cn.FINDER_REFERENCE_NAME].unique())
        prune_df = WorkerCheckpointManager.prune(df).pruned_df
        num_match = np.sum(prune_df[cn.FINDER_REFERENCE_NAME] == prune_df[cn.FINDER_TARGET_NAME])
        self.assertTrue(num_match >= num_unique)


if __name__ == '__main__':
    unittest.main(failfast=True)