from sirn.network_collection import NetworkCollection # type: ignore
from sirn.processed_network import ProcessedNetwork # type: ignore
from sirn.processed_network_collection import ProcessedNetworkCollection # type: ignore

import copy
import os
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
class TestClusteredNetworkCollection(unittest.TestCase):

    def setUp(self):
        network_collection = copy.deepcopy(NETWORK_COLLECTION.networks)
        clustered_networks = [ProcessedNetwork(n) for n in network_collection]
        self.clustered_network_collection = ProcessedNetworkCollection(clustered_networks,
                                                                       hash_val=HASH_VAL)
        collection = NetworkCollection.makeRandomCollection(num_network=1)
        self.other_clustered_network = ProcessedNetwork(collection.networks[0])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.clustered_network_collection, ProcessedNetworkCollection))
        self.assertEqual(len(self.clustered_network_collection), COLLECTION_SIZE)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_collection = self.clustered_network_collection.copy()
        self.assertEqual(self.clustered_network_collection, copy_collection)

    def testMakeFromRepr(self):
        if IGNORE_TEST:
            return
        repr_str = self.clustered_network_collection.__repr__()
        new_collection = self.clustered_network_collection.makeFromRepr(repr_str)
        self.assertEqual(self.clustered_network_collection, new_collection)

    def testRepr(self):
        if IGNORE_TEST:
            return
        repr_str = str(self.clustered_network_collection)
        self.assertTrue("+" in repr_str)

    def testAdd(self):
        if IGNORE_TEST:
            return
        current_len = len(self.clustered_network_collection)
        self.clustered_network_collection.add(self.other_clustered_network)
        self.assertEqual(current_len+1, len(self.clustered_network_collection))
    
    def testIsSubset(self):
        if IGNORE_TEST:
            return
        clustered_network_collection = self.clustered_network_collection.copy()
        self.assertTrue(self.clustered_network_collection.isSubset(clustered_network_collection))
        #
        clustered_network_collection.clustered_networks =  \
              clustered_network_collection.clustered_networks[1:]
        self.assertFalse(self.clustered_network_collection.isSubset(clustered_network_collection))


if __name__ == '__main__':
    unittest.main()