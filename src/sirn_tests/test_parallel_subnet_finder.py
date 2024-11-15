'''
1. Not selecting target correctly
2. Need to clean up created files
3. Is dependency injection clean
'''

import sirn.constants as cn  # type: ignore
from sirn.parallel_subnet_finder_worker import _CheckpointManager
from src.sirn.parallel_subnet_finder import ParallelSubnetFinder # type: ignore
from sirn.network import Network  # type: ignore

import json
import os
import pandas as pd # type: ignore
import numpy as np
import shutil
import unittest

IGNORE_TEST = False
IS_PLOT =  False
SIZE = 3
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")
CHECKPOINT_PATH = os.path.join(cn.DATA_DIR, "test_subnet_finder_checkpoint.csv")
REMOVE_FILES = [CHECKPOINT_PATH]


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
class TestFunctions(unittest.TestCase):



#############################
class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.checkpoint_manager = _CheckpointManager(CHECKPOINT_PATH, is_report=IS_PLOT)

    def tearDown(self):
        self.remove()

    def remove(self):
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)

    def testRecover(self):
        if IGNORE_TEST:
            return
        num_network = 10
        df = makeDataframe(num_network)
        df.loc[0, cn.FINDER_REFERENCE_NETWORK] = ""
        df.loc[0, cn.FINDER_INDUCED_NETWORK] = ""
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
        df.loc[0, cn.FINDER_REFERENCE_NETWORK] = ""
        df.loc[0, cn.FINDER_INDUCED_NETWORK] = ""
        df, deleteds = self.checkpoint_manager.prune(df)
        self.assertEqual(len(deleteds), 1)
        self.assertEqual(len(df), num_network - 1)


#############################
class TestParallelSubnetFinder(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.reference = Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
        self.target = self.reference.fill(num_fill_reaction=SIZE, num_fill_species=SIZE)
        self.finder = ParallelSubnetFinder(reference_networks=[self.reference], target_networks=[self.target],
              data_dir=cn.TEST_DIR, identity=cn.ID_WEAK)

    def tearDown(self):
        self.remove()

    def remove(self):
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)
        
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
        finder = ParallelSubnetFinder(reference_networks=reference_models, target_networks=target_models, identity=cn.ID_STRONG,
              data_dir=cn.DATA_DIR)
        df = finder.find(is_report=IS_PLOT)
        prune_df, _ = _prune(df)
        self.assertEqual(len(prune_df), NUM_REFERENCE_MODEL)
    
    def testFindFromDirectories(self):
        if IGNORE_TEST:
            return
        df = SubnetFinder.findFromDirectories(MODEL_DIR, MODEL_DIR, identity=cn.ID_WEAK, is_report=IS_PLOT,
              data_dir=cn.DATA_DIR)
        prune_df, _ = _prune(df)
        self.assertTrue(np.all(prune_df.reference_model == prune_df.target_model))

    def testFindBiomodelsSubnetSimple(self):
        if IGNORE_TEST:
            return
        df = SubnetFinder.findBiomodelsSubnet(max_num_target_network=10, reference_network_size=1,
              reference_network_names=["BIOMD0000000191"], is_report=IS_PLOT,
              identity=cn.ID_STRONG)
        prune_df, _ = _prune(df)
        self.assertEqual(len(prune_df), 1)

    # FIXME: Verify running 10 processes
    def testFindBiomodelsSubnetMultiplebatch(self):
        #if IGNORE_TEST:
        #    return
        df = SubnetFinder.findBiomodelsSubnet(reference_network_size=8,
              batch_size=1, is_initialize=True, is_report=IS_PLOT)
        df, processed_list = _prune(df)
        import pdb; pdb.set_trace()

    def serializeReferenceTargetNetworksOverlap(self, num_network:int=10, reference_size:int=5,
            target_fill_size:int=3, num_task:int=3):
        # Overlapping networks
        reference_networks = [Network.makeRandomNetworkByReactionType(reference_size,
              is_prune_species=True) for _ in range(num_network)]
        target_networks = [n.fill(num_fill_reaction=target_fill_size, num_fill_species=target_fill_size)
                for n in reference_networks]
        SubnetFinder._makeReferenceTargetSerializations(reference_networks, target_networks, num_task=num_task)

    def serializeReferenceTargetNetworksDisjoint(self, num_network:int=10, num_task:int=3):
        # Disjoint reference and target networks
        reference_networks = [Network.makeRandomNetworkByReactionType(5, is_prune_species=True) for _ in range(num_network)]
        target_networks = [Network.makeRandomNetworkByReactionType(5, is_prune_species=True) for _ in range(num_network)]
        SubnetFinder._makeReferenceTargetSerializations(reference_networks, target_networks, num_task=num_task)

    def testMakeReferenceTargetSerializations(self):
        if IGNORE_TEST:
            return
        num_task = 3
        self.serializeReferenceTargetNetworksDisjoint(num_task=num_task)
        ffiles = os.listdir(cn.DATA_DIR)
        self.assertEqual(len([f for f in ffiles if f.count("reference")>0]), num_task)
        self.assertEqual(len([f for f in ffiles if f.count("target")>0]), 1)

    def testExecuteTask(self):
        if IGNORE_TEST:
            return
        task_idx = 0
        num_task = 2
        num_network = 10
        self.serializeReferenceTargetNetworksDisjoint(num_task=num_task, num_network=num_network)
        df = self.finder._executeTask(task_idx, num_task, is_report=IS_PLOT, identity=cn.ID_STRONG,
          batch_size=1, is_initialize=True)
        self.assertEqual(len(df), 1/2*num_network**2)
        num_reference_network = len(df[cn.FINDER_REFERENCE_NAME].unique())
        self.assertEqual(num_reference_network, num_network/num_task)
        # Check for case that there are matches
        self.serializeReferenceTargetNetworksOverlap(num_task=num_task, num_network=num_network)
        df = self.finder._executeTask(task_idx, num_task, is_report=IS_PLOT, identity=cn.ID_STRONG,
          batch_size=1, is_initialize=True)
        expected_num_match = num_network/num_task
        actual_num_match = np.sum(df['is_truncated'] == False)
        self.assertEqual(actual_num_match, expected_num_match)



if __name__ == '__main__':
    unittest.main(failfast=False)