import sirn.constants as cn  # type: ignore
from src.sirn.subnet_finder import SubnetFinder, REFERENCE_NETWORK, REFERENCE_NAME, TARGET_NAME,  \
    INDUCED_NETWORK, NAME_DCT  # type: ignore
from src.sirn.parallel_subnet_finder import _CheckpointManager  # type: ignore
from sirn.network import Network  # type: ignore

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
    dct = {REFERENCE_NETWORK: [str(n) for n in range(num_network)],
           INDUCED_NETWORK: [str(n) for n in range(num_network, 2*num_network)]}
    df = pd.DataFrame(dct)
    df[REFERENCE_NAME] = [str(n) for n in reference_networks]
    df[TARGET_NAME] = [str(n) for n in target_networks]
    df[NAME_DCT] = [json.dumps(dict(a=n)) for n in range(num_network)]
    return df


#############################
class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.checkpoint_manager = _CheckpointManager(CHECKPOINT_PATH, is_report=IS_PLOT)

    def remove(self):
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    def tearDown(self):
        self.remove()

    def testRecover(self):
        if IGNORE_TEST:
            return
        num_network = 10
        df = makeDataframe(num_network)
        df.loc[0, REFERENCE_NETWORK] = ""
        df.loc[0, INDUCED_NETWORK] = ""
        self.checkpoint_manager.checkpoint(df)
        full_df, pruned_df, deleteds = self.checkpoint_manager.recover()
        self.assertEqual(len(full_df), num_network)
        self.assertEqual(len(pruned_df), num_network-1)
        self.assertEqual(len(deleteds), 1)

    def testPrune(self):
        if IGNORE_TEST:
            return
        num_network = 10
        df = makeDataframe(num_network)
        df.loc[0, REFERENCE_NETWORK] = ""
        df.loc[0, INDUCED_NETWORK] = ""
        df, deleteds = self.checkpoint_manager.prune(df)
        self.assertEqual(len(deleteds), 1)
        self.assertEqual(len(df), num_network - 1)


#############################
class TestSubnetFinder(unittest.TestCase):

    def setUp(self):
        self.reference = Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
        self.target = self.reference.fill(num_fill_reaction=SIZE, num_fill_species=SIZE)
        self.finder = SubnetFinder(reference_networks=[self.reference], target_networks=[self.target],
              identity=cn.ID_WEAK)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.finder.reference_networks), 0)

    def testFindSimple(self):
        if IGNORE_TEST:
            return
        df = self.finder.find(is_report=IS_PLOT)
        self.assertEqual(len(df), 1)

    def testFindScale(self):
        if IGNORE_TEST:
            return
        NUM_REFERENCE_MODEL = 100
        NUM_EXTRA_TARGET_MODEL = 100
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
        finder = SubnetFinder(reference_networks=reference_models, target_networks=target_models,
              identity=cn.ID_STRONG)
        df = finder.find(is_report=IS_PLOT)
        prune_df, _ = _CheckpointManager.prune(df)
        self.assertEqual(len(prune_df), NUM_REFERENCE_MODEL)
    
    def testFindFromDirectories(self):
        if IGNORE_TEST:
            return
        df = SubnetFinder.findFromDirectories(MODEL_DIR, MODEL_DIR, identity=cn.ID_WEAK, is_report=IS_PLOT)
        prune_df, _ = _CheckpointManager.prune(df)
        self.assertTrue(np.all(prune_df[REFERENCE_NAME] == prune_df[TARGET_NAME]))


if __name__ == '__main__':
    unittest.main(failfast=False)