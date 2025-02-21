from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB.network import Network # type: ignore
from pySubnetSB.matrix import Matrix # type: ignore
from pySubnetSB.network_collection import NetworkCollection # type: ignore
from pySubnetSB.processed_network import ProcessedNetwork # type: ignore
from pySubnetSB.cluster_builder import ClusterBuilder # type: ignore

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
        """ array1 = np.random.randint(0, 3, size=(num_species, num_reaction))
        array2 = np.random.randint(0, 3, size=(num_species, num_reaction))
        network = Network(array1, array2) """
        network = Network.makeRandomNetworkByReactionType(num_species=num_species, num_reaction=num_reaction)
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
            is_success = True
            for _ in range(5):  # Do multiple times to handle edge cases
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
                num_builder_collection = len(builder.processed_network_collections)
                if num_builder_collection >  num_collection:
                    num_indeterminate = np.sum([b.in_determinate for b in builder.processed_network_collections])
                    if num_indeterminate == num_builder_collection - num_collection:
                        is_success = True
                else:
                    if num_builder_collection == num_collection:
                        is_success = True
                for network_collection in network_collections:
                    self.assertTrue(str(network_collection) in str(network_collections))
                if is_success:
                    break
            self.assertTrue(is_success)
        #
        #test(num_collection=5, num_network=1000, num_species=15, num_reaction=15)
        test(num_species=4, num_reaction=4)
        test(num_collection=5, num_network=10)
        test(num_collection=5)
        test(num_collection=15, identity=cn.ID_WEAK)
            
    def testMakeNetworkCollection(self):
        if IGNORE_TEST:
            return
        ARRAY_SIZE = 5
        network_collection = NetworkCollection.makeRandomCollection(num_species=ARRAY_SIZE,
              num_reaction=ARRAY_SIZE, num_network=COLLECTION_SIZE)
        processed_networks = [ProcessedNetwork(network) for network in network_collection.networks]
        builder = ClusterBuilder(network_collection, is_report=IS_PLOT)
        for idx, processed_network in enumerate(processed_networks):
            network = builder.makeNetworkFromProcessedNetwork(processed_network)
            self.assertTrue(network == network_collection.networks[idx])

if __name__ == '__main__':
    unittest.main()