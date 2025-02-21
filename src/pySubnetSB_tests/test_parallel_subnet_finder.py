'''
1. Not selecting target correctly
2. Need to clean up created files
3. Is dependency injection clean
'''

import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.parallel_subnet_finder import ParallelSubnetFinder # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.parallel_subnet_finder_worker import WorkerCheckpointManager  # type: ignore
import pySubnetSB.util as util # type: ignore

#util.IS_TIMEIT = True

import os
import unittest

IGNORE_TEST = False
IS_PLOT =  False
SIZE = 10
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")
WORKUNIT_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subset_finder_workunit.csv")
BIOMODELS_DIR = os.path.join(cn.TEST_DIR, "xml_files")
BIOMODELS_SERIALIZATION_PATH = os.path.join(BIOMODELS_DIR, "biomodels_serialized.txt")
CHECKPOINT_PATH = os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_checkpoint.csv")
REFERENCE_SERIALIZATION_FILENAME = "test_parallel_subnet_finder_reference.txt"
TARGET_SERIALIZATION_FILENAME = "test_parallel_subnet_finder_target.txt"
REFERENCE_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, REFERENCE_SERIALIZATION_FILENAME)
TARGET_SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, TARGET_SERIALIZATION_FILENAME)
REMOVE_FILES = [CHECKPOINT_PATH, REFERENCE_SERIALIZATION_PATH, TARGET_SERIALIZATION_PATH, WORKUNIT_PATH]
REMOVE_FILES.extend([os.path.join(cn.TEST_DIR, "test_parallel_subnet_finder_checkpoint_%d.csv" % n)
      for n in range(10)])
NUM_NETWORK = 10


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
              TARGET_SERIALIZATION_PATH, workunit_csv_path=WORKUNIT_PATH, identity = cn.ID_STRONG,
              checkpoint_path=CHECKPOINT_PATH)

    def makeSerialization(sef, path, networks):
        serializer = ModelSerializer(None, path)
        serializer.serializeNetworks(networks)

    def tearDown(self):
        self.remove()

    def remove(self):
        WorkerCheckpointManager.deleteWorkerCheckpoints(CHECKPOINT_PATH, 10, is_report=IS_PLOT)
        manager = WorkerCheckpointManager(CHECKPOINT_PATH, is_report=IS_PLOT)
        manager.remove()
        for ffile in REMOVE_FILES:
            if os.path.exists(ffile):
                os.remove(ffile)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(os.path.exists(REFERENCE_SERIALIZATION_PATH))
        self.assertTrue(os.path.exists(TARGET_SERIALIZATION_PATH))
        self.assertGreater(len(self.finder.reference_networks), 0)

    @util.timeit
    def testFindOneProcess(self):
        if IGNORE_TEST:
            return
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, num_worker=1)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune_df = WorkerCheckpointManager.prune(df)[0]
        self.assertGreaterEqual(len(prune_df), NUM_NETWORK)
    
    @util.timeit
    def testFindOneProcessWithCheckpoint(self):
        if IGNORE_TEST:
            return
        _ = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, num_worker=1)
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=False, num_worker=1)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune_df = WorkerCheckpointManager.prune(df)[0]
        self.assertGreaterEqual(len(prune_df), NUM_NETWORK)
    
    @util.timeit
    def testFindManyProcess(self):
        if IGNORE_TEST:
            return
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, num_worker=2,
              max_num_assignment=1e9)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune_df = WorkerCheckpointManager.prune(df).pruned_df
        self.assertEqual(len(prune_df), NUM_NETWORK)
        #
        df = self.finder.parallelFind(is_report=IS_PLOT, is_initialize=True, num_worker=-1,
              max_num_assignment=1e1)
        self.assertEqual(len(df), NUM_NETWORK**2)
        prune2_df = WorkerCheckpointManager.prune(df).pruned_df
        self.assertLessEqual(len(prune2_df), len(prune_df))

    @util.timeit
    def testFindScale(self):
        if IGNORE_TEST:
            return
        NUM_REFERENCE_MODEL = 10
        NUM_EXTRA_TARGET_MODEL = 10
        NETWORK_SIZE = 10
        fill_size = 10
        self.remove()
        # Construct the models
        reference_models = [Network.makeRandomNetworkByReactionType(NETWORK_SIZE, is_prune_species=True)
              for _ in range(NUM_REFERENCE_MODEL)]
        target_models = [r.fill(num_fill_reaction=fill_size, num_fill_species=fill_size) for r in reference_models]
        # Add extra target models
        target_models.extend([Network.makeRandomNetworkByReactionType(NETWORK_SIZE, is_prune_species=True)
              for _ in range(NUM_EXTRA_TARGET_MODEL)])
        # Do the search
        df = ParallelSubnetFinder.findFromNetworks(
              reference_models,
              target_models,
              num_worker=10,
              identity=cn.ID_STRONG,
              serialization_dir=cn.TEST_DIR,
              is_report=IS_PLOT,
              is_initialize=True,
              checkpoint_path=CHECKPOINT_PATH,
              reference_serialization_filename=REFERENCE_SERIALIZATION_FILENAME,
              target_serialization_filename=TARGET_SERIALIZATION_FILENAME)
        prune_result = WorkerCheckpointManager.prune(df)
        self.assertLessEqual(len(prune_result.pruned_df), NUM_REFERENCE_MODEL)
    
    @util.timeit
    def testDirectoryFind(self):
        if IGNORE_TEST:
            return
        ffiles = [f for f in os.listdir(BIOMODELS_DIR) if f.endswith(".xml")]
        df = ParallelSubnetFinder.directoryFind(BIOMODELS_DIR,
              BIOMODELS_DIR, identity=cn.ID_WEAK,
              is_report=IS_PLOT, is_initialize=True, 
              checkpoint_path=CHECKPOINT_PATH,
              max_num_assignment=1e2)
        count = len(df[df[cn.FINDER_REFERENCE_NAME] == df[cn.FINDER_TARGET_NAME]])
        self.assertEqual(count, len(ffiles))

    @util.timeit
    def testBiomodelsFindSimple(self):
        if IGNORE_TEST:
            return
        self.remove()
        max_num_network = 2
        df = ParallelSubnetFinder.biomodelsFind(
              max_num_reference_network=max_num_network,
              max_num_target_network=max_num_network,
              reference_network_size=5,
              serialization_dir=cn.TEST_DIR,
              reference_serialization_filename=REFERENCE_SERIALIZATION_FILENAME,
              target_serialization_filename=TARGET_SERIALIZATION_FILENAME,
              is_report=IS_PLOT,
              checkpoint_path=CHECKPOINT_PATH,
              identity=cn.ID_STRONG)
        prune_result = WorkerCheckpointManager.prune(df)
        self.assertGreaterEqual(len(prune_result.full_df), max_num_network**2)


if __name__ == '__main__':
    unittest.main(failfast=True)