from sirn.network_collection import NetworkCollection # type: ignore
from sirn.clustered_network import ClusteredNetwork # type: ignore
from sirn.clustered_network_collection import ClusteredNetworkCollection # type: ignore
import sirn.constants as cn  # type: ignore

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
class TestClusteredNetworkCollection(unittest.TestCase):

    def setUp(self):
        network_collection = copy.deepcopy(NETWORK_COLLECTION.networks)
        self.clustered_networks = [ClusteredNetwork(n.network_name) for n in network_collection]
        self.clustered_network_collection = ClusteredNetworkCollection(self.clustered_networks,
                                                                       hash_val=HASH_VAL)
        collection = NetworkCollection.makeRandomCollection(num_network=1)
        self.other_clustered_network = ClusteredNetwork(collection.networks[0])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.clustered_network_collection, ClusteredNetworkCollection))
        self.assertEqual(len(self.clustered_network_collection), COLLECTION_SIZE)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_collection = self.clustered_network_collection.copy()
        self.assertEqual(self.clustered_network_collection, copy_collection)

    def testMakeFromRepr(self):
        if IGNORE_TEST:
            return
        for identity in [cn.ID_WEAK, cn.ID_STRONG]:
            clustered_network_collection = ClusteredNetworkCollection(self.clustered_networks,
                 hash_val=HASH_VAL, identity=identity)
            repr_str = clustered_network_collection.__repr__()
            new_collection = clustered_network_collection.makeFromRepr(repr_str)
            self.assertEqual(clustered_network_collection, new_collection)

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