import pySubnetSB.constants as cn  # type: ignore
import pySubnetSB.parallel_subnet_finder_worker as psfw  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.mock_queue import MockQueue  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.parallel_subnet_finder_worker import SubnetFinderWorkunitManager  # type: ignore
from pySubnetSB.worker_checkpoint_manager import WorkerCheckpointManager  # type: ignore

import json
import os
import pandas as pd # type: ignore
import numpy as np
from typing import Tuple
import unittest

IGNORE_TEST = False
IS_PLOT =  False
SIZE = 10
NUM_NETWORK = 10
TOTAL_WORKER = 2
BASE_CHECKPOINT_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_worker_checkpoint.csv")
WORKUNIT_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_worker_manager.csv")
TASK0_CHECKPOINT_PATH = psfw.WorkerCheckpointManager.makeWorkerCheckpointPath(BASE_CHECKPOINT_PATH, 0)
TASK1_CHECKPOINT_PATH = psfw.WorkerCheckpointManager.makeWorkerCheckpointPath(BASE_CHECKPOINT_PATH, 1)
TASK_CHECKPOINT_PATHS = [TASK0_CHECKPOINT_PATH, TASK1_CHECKPOINT_PATH]
REFERENCE_SERIALIZER_PATH = os.path.join(cn.TEST_DIR,
      "test_parallel_subnet_finder_worker_references.txt")
TARGET_SERIALIZER_PATH = os.path.join(cn.TEST_DIR,
      "test_parallel_subnet_finder_worker_targets.txt")
REMOVE_FILES:list = [BASE_CHECKPOINT_PATH, TASK0_CHECKPOINT_PATH, TASK1_CHECKPOINT_PATH,
      REFERENCE_SERIALIZER_PATH, TARGET_SERIALIZER_PATH, WORKUNIT_PATH]
REFERENCE_NETWORKS = [Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
      for _ in range(NUM_NETWORK)]
TARGET_NETWORKS = [n.fill(num_fill_reaction=SIZE, num_fill_species=SIZE) for n in REFERENCE_NETWORKS]


#############################
# Tests
#############################
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
    return df


#############################
class TestParallelSubnetFinderWorker(unittest.TestCase):

    def setUp(self):
        self.initializeNetworks()
        self.initializeWorkunitManager()

    def tearDown(self):
        self.remove()

    def remove(self):
        if hasattr(self, "manager"):
            self.manager.remove()
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)

    def initializeNetworks(self, num_network:int=NUM_NETWORK):
        reference_networks = [Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
                  for _ in range(num_network)]
        target_networks = [n.fill(num_fill_reaction=SIZE, num_fill_species=SIZE) for n in reference_networks]
        ModelSerializer.serializerFromNetworks(reference_networks, REFERENCE_SERIALIZER_PATH,
                is_initialize=True)
        ModelSerializer.serializerFromNetworks(target_networks, TARGET_SERIALIZER_PATH,
                is_initialize=True)

    def initializeWorkunitManager(self, total_worker:int=TOTAL_WORKER):
        if hasattr(self, "manager"):
            self.manager.remove()  # type: ignore
        self.manager = SubnetFinderWorkunitManager.makeFromSerializationFiles(WORKUNIT_PATH,
              num_worker=total_worker, reference_serialization_path=REFERENCE_SERIALIZER_PATH,
              target_serialization_path=TARGET_SERIALIZER_PATH)
        self.manager.makeWorkunitFile()

    def testExecuteTask(self):
        #if IGNORE_TEST:
        #    return
        #####
        def test(task_idx:int=0, is_initialize:bool=True, total_worker:int=2)->Tuple[pd.DataFrame, pd.DataFrame]:
            # Creates reference and target networks to assess processing by tasks.
            # Returns the full dataframe and the pruned dataframes
            psfw.executeWorker(task_idx, WORKUNIT_PATH, total_worker, BASE_CHECKPOINT_PATH,
                REFERENCE_SERIALIZER_PATH, TARGET_SERIALIZER_PATH, identity=cn.ID_STRONG,
                is_report=IS_PLOT, is_initialize=is_initialize, is_allow_exit=False)
            checkpoint_manager = psfw.WorkerCheckpointManager(TASK_CHECKPOINT_PATHS[task_idx],
                    is_report=IS_PLOT, is_initialize=False)
            full_df = checkpoint_manager.recover().full_df
            if len(full_df) == 0:
                return full_df, full_df
            prune_result = psfw.WorkerCheckpointManager.prune(full_df)
            return prune_result.full_df, prune_result.pruned_df
        #####
        # Finds the subnets for a single task
        total_worker = 1
        self.initializeWorkunitManager(total_worker=total_worker)
        full_df, prune_df = test(task_idx=0, total_worker=total_worker, is_initialize=True)
        self.assertEqual(len(full_df), NUM_NETWORK**2/total_worker)
        self.assertLessEqual(np.abs(len(prune_df) - NUM_NETWORK/total_worker), 1)
        # Finds all subnets if there is an existing checkpoint
        # FIXME: Test for recovering a checkpoint
        total_worker = 1
        self.initializeWorkunitManager(total_worker=total_worker)
        _ = test(task_idx=0, is_initialize=True, total_worker=total_worker)
        full2_df, _ = test(task_idx=0, is_initialize=False, total_worker=total_worker)
        self.assertEqual(len(full2_df), NUM_NETWORK**2)
        # Handle two tasks
        total_worker = 2
        WorkerCheckpointManager.deleteWorkerCheckpoints(WORKUNIT_PATH, total_worker)
        self.initializeWorkunitManager(total_worker=total_worker)
        for task_idx in range(2):
            full_df, prune_df = test(task_idx=task_idx, total_worker=total_worker,
                  is_initialize=True)
            self.assertEqual(len(full_df), NUM_NETWORK**2/2)


if __name__ == '__main__':
    unittest.main(failfast=False)