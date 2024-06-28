from sirn import constants as cn  # type: ignore
from sirn.network import Network # type: ignore
from sirn.network_collection import NetworkCollection # type: ignore
from sirn.cluster_buildern import ClusterBuilder, ClusteredNetwork, ClusteredNetworkCollection # type: ignore

import copy
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = True
IS_PLOT = False
COLLECTION_SIZE = 10
NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=COLLECTION_SIZE)


#############################
# Tests
#############################
class TestNetworkCollection(unittest.TestCase):

    def setUp(self):
        self.collection = copy.deepcopy(NETWORK_COLLECTION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(len(self.collection) == COLLECTION_SIZE)

    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(str(self.collection), str))

    def testMakeRandomCollection(self):
        if IGNORE_TEST:
            return
        size = 10
        collection = NetworkCollection.makeRandomCollection(num_network=size)
        self.assertTrue(len(collection) == size)

    def makeStructurallyIdenticalCollection(self, num_network:int=5, num_row:int=5, num_column:int=7,
                                            structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_STRONG):
        array1 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        array2 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        network = Network(array1, array2)
        networks = [network.randomize(structural_identity_type=structural_identity_type)
                    for _ in range(num_network)]
        return NetworkCollection(networks, structural_identity_type=structural_identity_type)
    
    def testAdd(self):
        if IGNORE_TEST:
            return
        collection1 = self.makeStructurallyIdenticalCollection(num_network=15)
        collection2 = self.makeStructurallyIdenticalCollection()
        collection = collection1 + collection2
        self.assertTrue(len(collection) == len(collection1) + len(collection2))
        self.assertEqual(len(collection), str(collection).count("---") + 1)

    def testCluster(self):
        if IGNORE_TEST:
            return
        # Construct a collection of two sets of permutably identical matrices
        def test(num_collection=2, num_network=5, num_row=5, num_column=5,
                 structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_STRONG):
            # Make collection of structurally identical networks
            network_collections = [self.makeStructurallyIdenticalCollection(
                structural_identity_type=structural_identity_type,
                num_row=num_row, num_column=num_column, num_network=num_network)
                for _ in range(num_collection)]
            # Construct the pm_collection to analyze and is the combination of the other pm_collections
            network_collection = network_collections[0]
            for network in network_collections[1:]:
                network_collection += network
            network_collections = network_collection.cluster(is_report=False,
                    max_log_perm=2)
            import pdb; pdb.set_trace()
            self.assertEqual(len(network_collections), num_collection)
            for network_collection in network_collections:
                self.assertTrue(str(network_collection) in str(network_collections))
        #
        test(num_collection=5, num_network=10)
        test()
        test(num_collection=5)
        test(num_collection=15, structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_WEAK)

    def testMakeFromAntimonyDirectory(self):
        if IGNORE_TEST:
            return
        directory = os.path.join(cn.TEST_DIR, "oscillators")
        network_collection = NetworkCollection.makeFromAntimonyDirectory(directory)
        self.assertTrue(len(network_collection) > 0)

    def checkSerializeDeserialize(self, collection:NetworkCollection):
        df = collection.serialize()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(len(df) == len(collection))
        #
        new_collection = NetworkCollection.deserialize(df)
        self.assertTrue(isinstance(new_collection, NetworkCollection))
        self.assertTrue(collection == new_collection)
    
    def testSerializeDeserialize1(self):
        if IGNORE_TEST:
            return
        self.checkSerializeDeserialize(self.collection)

    def testSerializeDeserialize2(self):
        if IGNORE_TEST:
            return
        def test(num_network:int=5, array_size:int=5, is_structural_identity:bool=True):
            collection = NetworkCollection.makeRandomCollection(array_size=array_size,
                num_network=num_network)
            self.checkSerializeDeserialize(collection)
        #
        test(is_structural_identity=False)
        test(is_structural_identity=True)
        test(num_network=10, array_size=10)

    def testMaxLogPermutations(self):
        #if IGNORE_TEST:
        #    return
        # Construct a collection of two sets of permutably identical matrices
        def test(max_log_perm=2, num_collection=2, num_network=5, num_row=5, num_column=5,
                 structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_STRONG):
            # Make collection of structurally identical networks
            network_collections = [self.makeStructurallyIdenticalCollection(
                structural_identity_type=structural_identity_type,
                num_row=num_row, num_column=num_column, num_network=num_network)
                for _ in range(num_collection)]
            # Construct the pm_collection to analyze and is the combination of the other pm_collections
            network_collection = network_collections[0]
            for network in network_collections[1:]:
                network_collection += network
            network_collections = network_collection.cluster(is_report=False,
                    max_log_perm=max_log_perm)
            count_indeterminant_identities =  \
                np.sum([network_collection.structural_identity_type == cn.UNKNOWN_STRUCTURAL_IDENTITY_NAME
                        for network_collection in network_collections])
            return count_indeterminant_identities
        #
        for num_network in [5, 10, 15]:
            count1 = test(num_network=num_network, max_log_perm=1, num_collection=10)
            count15 = test(num_network=num_network, max_log_perm=1.5, num_collection=10)
            count2 = test(num_network=num_network, max_log_perm=2, num_collection=10)
            self.assertGreaterEqual(count1, count15)
            self.assertGreaterEqual(count2, count15)


if __name__ == '__main__':
    unittest.main()