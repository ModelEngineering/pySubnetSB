from pySubnetSB.network_collection import NetworkCollection # type: ignore
from pySubnetSB.processed_network import ProcessedNetwork # type: ignore

import copy
import unittest


IGNORE_TEST = False
IS_PLOT = False
NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=1)


#############################
# Tests
#############################
class TestprocessedNetwork(unittest.TestCase):

    def setUp(self):
        network = copy.deepcopy(NETWORK_COLLECTION.networks[0])
        self.processed_network = ProcessedNetwork(network.network_name)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.processed_network, ProcessedNetwork))

    def testCopy(self):
        if IGNORE_TEST:
            return
        copy_network = self.processed_network.copy()
        self.assertTrue(copy_network == self.processed_network)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        for is_indeterminate in [False, True]:
            self.processed_network.setIndeterminate(is_indeterminate)
            serialization_str = self.processed_network.serialize()
            processed_network = self.processed_network.deserialize(serialization_str)
            self.assertEqual(self.processed_network, processed_network)


if __name__ == '__main__':
    unittest.main()