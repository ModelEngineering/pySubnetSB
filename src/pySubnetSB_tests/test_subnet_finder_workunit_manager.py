import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.subnet_finder_workunit_manager import SubnetFinderWorkunitManager # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore

import os
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
WORKUNIT_PATH = os.path.join(cn.TEST_DIR, "test_subnet_finder_workunit_manager.csv")
TARGET_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, "test_subnet_finder_workunit_manager_targets.txt")
REFERENCE_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR,
      "test_subnet_finder_workunit_manager_references.txt")
NUM_REFERENCE_NETWORK = 5
NUM_TARGET_NETWORK = 6
NUM_WORKER = 10
REFERENCE_NETWORKS = [Network.makeRandomNetworkByReactionType(10, is_prune_species=True)
      for _ in range(NUM_REFERENCE_NETWORK)]
TARGET_NETWORKS = [Network.makeRandomNetworkByReactionType(10, is_prune_species=True)
      for _ in range(NUM_TARGET_NETWORK)]


#############################
# Tests
#############################
class TestSubnetFinderWorkunitManager(unittest.TestCase):

    def setUp(self):
        self.manager = SubnetFinderWorkunitManager(WORKUNIT_PATH,
              num_worker=NUM_WORKER, reference_networks=REFERENCE_NETWORKS,
                target_networks=TARGET_NETWORKS)
        self.num_workunit = NUM_REFERENCE_NETWORK * NUM_TARGET_NETWORK
        self.workunit_per_worker = self.num_workunit / NUM_WORKER
        self.manager.remove()

    def tearDown(self):
        self.manager.remove()
        for serialization_path in [TARGET_SERIALIZATION_PATH, REFERENCE_SERIALIZATION_PATH]:
            serializer = ModelSerializer(None, serialization_path)
            serializer.remove()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(NUM_WORKER, self.manager.num_worker)
        self.assertEqual(NUM_REFERENCE_NETWORK, self.manager.num_reference_network)
        self.assertEqual(NUM_TARGET_NETWORK, self.manager.num_target_network)
        self.assertEqual(WORKUNIT_PATH, self.manager.csv_path)

    def testMakeWorkunitFile(self):
        if IGNORE_TEST:
            return
        serializer = ModelSerializer(None, serialization_path=REFERENCE_SERIALIZATION_PATH)
        serializer.serializeNetworks(REFERENCE_NETWORKS)
        serializer = ModelSerializer(None, serialization_path=TARGET_SERIALIZATION_PATH)
        serializer.serializeNetworks(TARGET_NETWORKS)
        self.manager.makeWorkunitFile()
        df = self.manager.getWorkunitDataframe()
        self.assertEqual(NUM_REFERENCE_NETWORK*NUM_TARGET_NETWORK, len(df))
        for worker_idx in range(NUM_WORKER):
            count = len(df[df[cn.FINDER_WORKER_IDX] == worker_idx])
            self.assertTrue(count >= self.workunit_per_worker-1)
    
    def testMakeWorkunitFileFromPath(self):
        if IGNORE_TEST:
            return
        self.manager.makeWorkunitFile()
        df = self.manager.getWorkunitDataframe()
        self.assertEqual(NUM_REFERENCE_NETWORK*NUM_TARGET_NETWORK, len(df))
        for worker_idx in range(NUM_WORKER):
            count = len(df[df[cn.FINDER_WORKER_IDX] == worker_idx])
            self.assertTrue(count >= self.workunit_per_worker-1)

    def testMakeWorkunitFile2(self):
        if IGNORE_TEST:
            return
        self.manager.makeWorkunitFile()
        worker_df = self.manager.getWorkunitDataframe()
        # Ensure that all workunits are present and unique
        for reference_idx in range(NUM_REFERENCE_NETWORK):
            for target_idx in range(NUM_TARGET_NETWORK):
                workunit = worker_df[(worker_df[cn.FINDER_REFERENCE_IDX] == reference_idx) &
                      (worker_df[cn.FINDER_TARGET_IDX] == target_idx)]
                self.assertTrue(len(workunit) == 1)

    def testGetWorkunits(self):
        if IGNORE_TEST:
            return
        self.manager.makeWorkunitFile()
        for worker_idx in range(NUM_WORKER):
            workunits = self.manager.getWorkunits(worker_idx)
            self.assertTrue(np.abs(self.workunit_per_worker - len(workunits)) <= 1)

    def testGetWorkunitsPriorWork(self):
        if IGNORE_TEST:
            return
        num_processed = 2   # Delete some workunits. They should be recovered.
        worker_idx = 0
        self.manager.makeWorkunitFile()
        worker_df = self.manager.getWorkunitDataframe(worker_idx=worker_idx)
        processed_reference_networks = [worker_df.iloc[i][cn.FINDER_REFERENCE_NAME]
              for i in range(num_processed)]   
        processed_target_networks = [worker_df.iloc[i][cn.FINDER_TARGET_NAME]
              for i in range(num_processed)]   
        # Delete some workunits
        # Should just have the deleted workunits
        workunits = self.manager.getWorkunits(worker_idx,
              processed_target_networks=processed_target_networks,
              processed_reference_networks=processed_reference_networks)
        self.assertEqual(len(workunits), len(worker_df) - num_processed)

    def testRemove(self):
        if IGNORE_TEST:
            return
        # Smoke test
        self.manager.remove()
        # Create file
        self.manager.makeWorkunitFile()
        self.manager.remove()
        self.assertFalse(os.path.exists(WORKUNIT_PATH))


if __name__ == '__main__':
    unittest.main()