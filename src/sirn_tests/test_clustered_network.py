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
        self.clustered_network = ClusteredNetwork(network.network_name)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.clustered_network, ClusteredNetwork))

    def testCopy(self):
        if IGNORE_TEST:
            return
        copy_network = self.clustered_network.copy()
        self.assertTrue(copy_network == self.clustered_network)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        for is_indeterminate in [False, True]:
            self.clustered_network.setIndeterminate(is_indeterminate)
            serialization_str = self.clustered_network.serialize()
            clustered_network = self.clustered_network.deserialize(serialization_str)
            self.assertEqual(self.clustered_network, clustered_network)


if __name__ == '__main__':
    unittest.main()