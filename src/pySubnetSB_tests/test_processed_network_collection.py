from pySubnetSB.network_collection import NetworkCollection # type: ignore
from pySubnetSB.processed_network import ProcessedNetwork # type: ignore
from pySubnetSB.processed_network_collection import ProcessedNetworkCollection # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import copy
import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
HASH_VAL = 1111
COLLECTION_SIZE = 50
NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=COLLECTION_SIZE)


#############################
# Tests
#############################
class TestProcessedNetworkCollection(unittest.TestCase):

    def setUp(self):
        network_collection = copy.deepcopy(NETWORK_COLLECTION.networks)
        self.processed_networks = [ProcessedNetwork(n.network_name) for n in network_collection]
        self.processed_network_collection = ProcessedNetworkCollection(self.processed_networks,
                                                                       hash_val=HASH_VAL)
        collection = NetworkCollection.makeRandomCollection(num_network=1)
        self.other_processed_network = ProcessedNetwork(collection.networks[0])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.processed_network_collection, ProcessedNetworkCollection))
        self.assertEqual(len(self.processed_network_collection), COLLECTION_SIZE)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_collection = self.processed_network_collection.copy()
        self.assertEqual(self.processed_network_collection, copy_collection)

    def testAdd(self):
        if IGNORE_TEST:
            return
        current_len = len(self.processed_network_collection)
        self.processed_network_collection.add(self.other_processed_network)
        self.assertEqual(current_len+1, len(self.processed_network_collection))
    
    def testIsSubset(self):
        if IGNORE_TEST:
            return
        processed_network_collection = self.processed_network_collection.copy()
        self.assertTrue(self.processed_network_collection.isSubset(processed_network_collection))
        #
        processed_network_collection.processed_networks =  \
              processed_network_collection.processed_networks[1:]
        self.assertFalse(self.processed_network_collection.isSubset(processed_network_collection))
    
    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        for identity in [cn.ID_WEAK, cn.ID_STRONG]:
            self.processed_network_collection.identity = identity
            serialization_str = self.processed_network_collection.serialize()
            processed_network_collection = self.processed_network_collection.deserialize(serialization_str)
            self.assertEqual(self.processed_network_collection, processed_network_collection)


if __name__ == '__main__':
    unittest.main()