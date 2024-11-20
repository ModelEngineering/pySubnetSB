'''
1. Not selecting target correctly
2. Need to clean up created files
3. Is dependency injection clean
'''

import sirn.constants as cn  # type: ignore
from src.sirn.parallel_subnet_finder import ParallelSubnetFinder # type: ignore
from sirn.network import Network  # type: ignore
from sirn.model_serializer import ModelSerializer  # type: ignore
from sirn.parallel_subnet_finder_worker import WorkerCheckpointManager

import os
import pandas as pd # type: ignore
import numpy as np
from typing import Optional
import unittest

IGNORE_TEST = True
IS_PLOT =  False
SIZE = 10
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")
CHECKPOINT_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_checkpoint.csv")
REFERENCE_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_reference.txt")
TARGET_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_target.txt")
REMOVE_FILES = [CHECKPOINT_PATH, REFERENCE_SERIALIZATION_PATH, TARGET_SERIALIZATION_PATH]
REMOVE_FILES.extend([os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_checkpoint_%d.csv" % n)
      for n in range(10)])
NUM_NETWORK = 100


#############################
class TestParallelSubnetFinder(unittest.TestCase):

    def setUp(self):
        self.remove()
        reference_networks = [Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
                for _ in range(NUM_NETWORK)]
        target_networks = [n.fill(num_fill_reaction=SIZE, num_fill_species=SIZE) for n in reference_networks]
        self.makeSerialization(REFERENCE_SERIALIZATION_PATH, reference_networks)
        self.makeSerialization(TARGET_SERIALIZATION_PATH, target_networks)
        self.finder = ParallelSubnetFinder(REFERENCE_SERIALIZATION_PATH,
              TARGET_SERIALIZATION_PATH, identity = cn.ID_STRONG,
              checkpoint_path=CHECKPOINT_PATH)

    def makeSerialization(sef, path, networks):
        serializer = ModelSerializer(None, path)
        serializer.serializeNetworks(networks)

    def tearDown(self):
        self.remove()

    def remove(self):
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        import pdb; pdb.set_trace()
        self.assertTrue(os.path.exists(REFERENCE_SERIALIZATION_PATH))
        self.assertTrue(os.path.exists(TARGET_SERIALIZATION_PATH))
        self.assertGreater(len(self.finder.reference_networks), 0)

    def testFindOneProcess(self):
        if IGNORE_TEST:
            return
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, total_process=1)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune_df = WorkerCheckpointManager.prune(df)[0]
        self.assertGreaterEqual(len(prune_df), NUM_NETWORK)
    
    def testFindManyProcess(self):
        #if IGNORE_TEST:
        #    return
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, total_process=-1,
              max_num_assignment=1e9)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune_df = WorkerCheckpointManager.prune(df)[0]
        self.assertEqual(len(prune_df), NUM_NETWORK)
        #
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, total_process=-1,
              max_num_assignment=1e1)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune2_df = WorkerCheckpointManager.prune(df)[0]
        self.assertLessEqual(len(prune2_df), len(prune_df))

    def testFindScale(self):
        if IGNORE_TEST:
            return
        NUM_REFERENCE_MODEL = 1000
        NUM_EXTRA_TARGET_MODEL = 1000
        NETWORK_SIZE = 10
        fill_size = 10
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
        prune_df = WorkerCheckpointManager.prune(df)
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
        if IGNORE_TEST:
            return
        df = SubnetFinder.findBiomodelsSubnet(reference_network_size=8,
              batch_size=1, is_initialize=True, is_report=IS_PLOT)
        df, processed_list = _prune(df)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    unittest.main(failfast=False)