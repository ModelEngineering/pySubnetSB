from sirn import constants as cn  # type: ignore
from sirn.network import Network # type: ignore
from sirn.pmatrix import PMatrix # type: ignore
from sirn.network_collection import NetworkCollection # type: ignore
from sirn.clustered_network import ClusteredNetwork # type: ignore
from sirn.cluster_builder import ClusterBuilder # type: ignore

import copy
import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
COLLECTION_SIZE = 10
if not IGNORE_TEST:
    NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=COLLECTION_SIZE)


#############################
# Tests
#############################
class TestClusterBuilder(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        network_collection = copy.deepcopy(NETWORK_COLLECTION)
        self.builder = ClusterBuilder(network_collection, is_report=IS_PLOT,
                                      max_num_perm=100, is_structural_identity_strong=True)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(len(self.builder.network_collection) == COLLECTION_SIZE)
        self.assertTrue(isinstance(self.builder.network_collection, NetworkCollection))

    def testMakeHashDct(self):
        if IGNORE_TEST:
            return
        hash_dct = self.builder._makeHashDct()
        count = np.sum([len(v) for v in hash_dct.values()])  
        self.assertTrue(count == COLLECTION_SIZE)
        #
        for networks in hash_dct.values():
            for network in networks:
                is_true = any([network == n for n in self.builder.network_collection.networks])
                self.assertTrue(is_true)

    def makeStructurallyIdenticalCollection(self, num_network:int=5, num_row:int=5, num_column:int=7,
                                            structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_STRONG):
        array1 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        array2 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        network = Network(array1, array2)
        networks = []
        for _ in range(num_network):
            new_network = network.randomize(structural_identity_type=structural_identity_type)
            networks.append(new_network)
        return NetworkCollection(networks)

    def makeStronglyIdenticalCollection(self, num_network:int=5, num_row:int=5, num_column:int=7):
        array1 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        array2 = PMatrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        network = Network(array1, array2)
        networks = []
        for _ in range(num_network):
            new_network = network.randomize(is_verify=False)
            networks.append(new_network)
        return NetworkCollection(networks)

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
            #
            builder = ClusterBuilder(network_collection, max_num_perm=100, is_report=IS_PLOT)
            builder.cluster()
            self.assertEqual(len(network_collections), num_collection)
            for network_collection in network_collections:
                self.assertTrue(str(network_collection) in str(network_collections))
        #
        test()
        test(num_collection=5, num_network=10)
        test(num_collection=5)
        test(num_collection=15, structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_WEAK)

    def checkSerializeDeserialize(self, cluster_builder:ClusterBuilder):
        cluster_builder.cluster()
        df = cluster_builder.serializeClusteredNetworkCollections()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(isinstance(df.attrs, dict))
        self.assertTrue(len(df) == len(cluster_builder.clustered_network_collections))
        #
        clustered_network_collections = ClusterBuilder.deserializeClusteredNetworkCollections(df)
        trues = [c1 == c2 for c1, c2 in zip(clustered_network_collections,
                                            cluster_builder.clustered_network_collections)]
        self.assertTrue(all(trues))
    
    def testSerializeDeserialize1(self):
        if IGNORE_TEST:
            return
        self.checkSerializeDeserialize(self.builder)

    def testSerializeDeserialize2(self):
        if IGNORE_TEST:
            return
        def test(num_network:int=5, array_size:int=5, is_structural_identity_strong:bool=True):
            collection = NetworkCollection.makeRandomCollection(array_size=array_size,
                num_network=num_network)
            cluster = ClusterBuilder(collection, is_report=IS_PLOT,
                                     is_structural_identity_strong=is_structural_identity_strong)
            self.checkSerializeDeserialize(cluster)
        #
        test(is_structural_identity_strong=False)
        test(is_structural_identity_strong=True)
        test(num_network=10, array_size=10)

    def testExceedPermutationCount(self):
        if IGNORE_TEST:
            return
        # Construct a collection of two sets of permutably identical matrices
        def test(max_num_perm=100, num_collection=2, num_network=5, num_row=10, num_column=10,
                 structural_identity_type=cn.STRUCTURAL_IDENTITY_TYPE_STRONG,
                 is_verify=False):
            # Make collection of structurally identical networks
            network_collections = []
            for _ in range(5):  # Avoid name collisions randomly
                for _ in range(num_collection):
                    if is_verify:
                        network_collection = self.makeStructurallyIdenticalCollection(
                            structural_identity_type=structural_identity_type,
                            num_row=num_row, num_column=num_column, num_network=num_network)
                    else:
                        network_collection = self.makeStronglyIdenticalCollection(
                            num_row=num_row, num_column=num_column, num_network=num_network)
                    network_collections.append(network_collection)
                # Construct the pm_collection to analyze and is the combination of the other pm_collections
                network_collection = network_collections[0]
                for new_network_collection in network_collections[1:]:
                    network_collection += new_network_collection
                builder = ClusterBuilder(network_collection, max_num_perm=max_num_perm,
                                        is_report=IS_PLOT)
                builder.cluster()
                count =  np.sum(["?" in str(c) for c in builder.clustered_network_collections]) 
                if count > 0:
                    self.assertTrue(True)
                    return
            import pdb; pdb.set_trace()
            self.assertTrue(False)
            
        #
        for num_network in [5, 10, 15]:
            test(num_network=num_network, max_num_perm=1, num_collection=10)
            test(num_network=num_network, max_num_perm=5, num_collection=10)
            
    def testMakeNetworkCollection(self):
        if IGNORE_TEST:
            return
        ARRAY_SIZE = 5
        network_collection = NetworkCollection.makeRandomCollection(array_size=ARRAY_SIZE,
              num_network=COLLECTION_SIZE)
        clustered_networks = [ClusteredNetwork(network) for network in network_collection.networks]
        builder = ClusterBuilder(network_collection, is_report=IS_PLOT)
        for idx, clustered_network in enumerate(clustered_networks):
            network = builder.makeNetworkFromClusteredNetwork(clustered_network)
            self.assertTrue(network == network_collection.networks[idx])

if __name__ == '__main__':
    unittest.main()