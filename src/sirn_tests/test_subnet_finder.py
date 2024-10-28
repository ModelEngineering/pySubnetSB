from src.sirn.subnet_finder import SubnetFinder # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn  # type: ignore

import os
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIZE = 3
REMOVE_DIRS:list = []
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")


#############################
# Tests
#############################
class TestSubnetFinder(unittest.TestCase):

    def setUp(self):
        self.reference = Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=True)
        self.target = self.reference.fill(num_fill_reaction=SIZE, num_fill_species=SIZE)
        self.finder = SubnetFinder(reference_models=[self.reference], target_models=[self.target], identity=cn.ID_WEAK)
        
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.finder.reference_models), 0)

    def testFindSimple(self):
        if IGNORE_TEST:
            return
        df = self.finder.find(is_report=IS_PLOT)
        self.assertEqual(len(df), 1)

    def testFindScale(self):
        if IGNORE_TEST:
            return
        NUM_REFERENCE_MODEL = 50
        NUM_EXTRA_TARGET_MODEL = 50
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
        finder = SubnetFinder(reference_models=reference_models, target_models=target_models, identity=cn.ID_STRONG)
        df = finder.find(is_report=IS_PLOT)
        self.assertEqual(len(df), NUM_REFERENCE_MODEL)
    
    def testFindFromDirectories(self):
        if IGNORE_TEST:
            return
        df = SubnetFinder.findFromDirectories(MODEL_DIR, MODEL_DIR, identity=cn.ID_WEAK, is_report=IS_PLOT)
        self.assertTrue(np.all(df.reference_model == df.target_model))


if __name__ == '__main__':
    unittest.main(failfast=True)