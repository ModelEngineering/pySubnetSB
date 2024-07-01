from sirn.network_collection import NetworkCollection # type: ignore
from sirn.clustered_network import ClusteredNetwork # type: ignore

import copy
import unittest


IGNORE_TEST = False
IS_PLOT = False
NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=1)


#############################
# Tests
#############################
class TestClusteredNetwork(unittest.TestCase):

    def setUp(self):
        network = copy.deepcopy(NETWORK_COLLECTION.networks[0])
        self.clustered_network = ClusteredNetwork(network)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.clustered_network, ClusteredNetwork))

    def testCopy(self):
        if IGNORE_TEST:
            return
        copy_network = self.clustered_network.copy()
        self.assertTrue(copy_network == self.clustered_network)

    def testAdd(self):
        if IGNORE_TEST:
            return
        self.clustered_network.add(5)
        self.assertTrue(self.clustered_network.num_perm == 5)

    def testParseMake(self):
        if IGNORE_TEST:
            return
        repr_str = self.clustered_network.__repr__()
        network = ClusteredNetwork.makeFromRepr(repr_str)
        self.assertTrue(network == self.clustered_network)


if __name__ == '__main__':
    unittest.main()