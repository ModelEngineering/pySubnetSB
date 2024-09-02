from sirn import constants as cn  # type: ignore
from sirn.network import Network # type: ignore
from sirn.matrix import Matrix # type: ignore
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
                                      max_num_assignment=100, identity=cn.ID_WEAK)

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

    def makeStructurallyIdenticalCollection(self, num_network:int=5, num_species:int=5, num_reaction:int=7):
        """Makes a structurally identical collection with strong identity.

        Args:
            num_network (int, optional): _description_. Defaults to 5.
            num_row (int, optional): _description_. Defaults to 5.
            num_column (int, optional): _description_. Defaults to 7.

        Returns:
            _type_: _description_
        """
        array1 = Matrix.makeTrinaryMatrix(num_row=num_species, num_column=num_reaction)
        array2 = Matrix.makeTrinaryMatrix(num_row=num_species, num_column=num_reaction)
        network = Network(array1, array2)
        networks = []
        for _ in range(num_network):
            new_network, _ = network.permute()
            networks.append(new_network)
        return NetworkCollection(networks)

    def makeStronglyIdenticalCollection(self, num_network:int=5, num_row:int=5, num_column:int=7):
        array1 = Matrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        array2 = Matrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        network = Network(array1, array2)
        networks = []
        for _ in range(num_network):
            new_network, _ = network.permute()
            networks.append(new_network)
        return NetworkCollection(networks)

    def testCluster(self):
        if IGNORE_TEST:
            return
        # Construct a collection of two sets of permutably identical matrices
        def test(num_collection=2, num_network=5, num_species=15, num_reaction=15,
                 identity=cn.ID_STRONG):
            # Make disjoint network collections, each of which is structurally identical
            network_collections = [self.makeStructurallyIdenticalCollection(
                num_species=num_species, num_reaction=num_reaction, num_network=num_network)
                for _ in range(num_collection)]
            # Construct the network_collection to analyze that it is the combination of the other network_collections
            network_collection = network_collections[0]
            for network in network_collections[1:]:
                try:
                    network_collection += network
                except ValueError:
                    # Duplicate randomly generated name. Ignore.
                    pass
            #
            builder = ClusterBuilder(network_collection, max_num_assignment=100000, is_report=IS_PLOT,
                                     identity=identity)
            builder.cluster()
            self.assertEqual(len(builder.clustered_network_collections), num_collection)
            for network_collection in network_collections:
                self.assertTrue(str(network_collection) in str(network_collections))
        #
        #test(num_collection=5, num_network=1000, num_species=15, num_reaction=15)
        test(num_collection=5, num_network=10)
        test()
        test(num_collection=5)
        test(num_collection=15, identity=cn.ID_WEAK)

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
        def test(num_network:int=5, size:int=5):
            collection = NetworkCollection.makeRandomCollection(num_species=size, num_reaction=size,
                num_network=num_network)
            cluster = ClusterBuilder(collection, is_report=IS_PLOT, identity=cn.ID_STRONG)
            self.checkSerializeDeserialize(cluster)
        #
        test()
        test(num_network=10, size=10)
            
    def testMakeNetworkCollection(self):
        if IGNORE_TEST:
            return
        ARRAY_SIZE = 5
        network_collection = NetworkCollection.makeRandomCollection(num_species=ARRAY_SIZE,
              num_reaction=ARRAY_SIZE, num_network=COLLECTION_SIZE)
        clustered_networks = [ClusteredNetwork(network) for network in network_collection.networks]
        builder = ClusterBuilder(network_collection, is_report=IS_PLOT)
        for idx, clustered_network in enumerate(clustered_networks):
            network = builder.makeNetworkFromClusteredNetwork(clustered_network)
            self.assertTrue(network == network_collection.networks[idx])

if __name__ == '__main__':
    unittest.main()