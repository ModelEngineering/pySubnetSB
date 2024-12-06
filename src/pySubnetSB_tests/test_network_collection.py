from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB.network import Network # type: ignore
from pySubnetSB.network_collection import NetworkCollection # type: ignore
from pySubnetSB.matrix import Matrix  # type: ignore

import copy
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
COLLECTION_SIZE = 10
NETWORK_COLLECTION = NetworkCollection.makeRandomCollection(num_network=COLLECTION_SIZE)
XML_DIR = os.path.join(cn.TEST_DIR, "xml_files")


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

    def makeStructurallyIdenticalCollection(self, num_network:int=5, num_row:int=5, num_column:int=7):
        array1 = Matrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        array2 = Matrix.makeTrinaryMatrix(num_row=num_row, num_column=num_column)
        network = Network(array1, array2)
        networks = [network.permute()[0] for _ in range(num_network)]
        return NetworkCollection(networks)
    
    def testAdd(self):
        if IGNORE_TEST:
            return
        collection1 = self.makeStructurallyIdenticalCollection(num_network=15)
        collection2 = self.makeStructurallyIdenticalCollection()
        collection = collection1 + collection2
        self.assertTrue(len(collection) == len(collection1) + len(collection2))
        self.assertEqual(len(collection), str(collection).count("---") + 1)

    def testMakeFromAntimonyDirectory(self):
        if IGNORE_TEST:
            return
        directory = os.path.join(cn.TEST_DIR, "oscillators")
        network_collection = NetworkCollection.makeFromAntimonyDirectory(directory)
        self.assertTrue(len(network_collection) > 0)

    def testMakeFromAntimonyDirectoryXML(self):
        if IGNORE_TEST:
            return
        network_collection = NetworkCollection.makeFromAntimonyDirectory(XML_DIR, batch_size=-1)
        self.assertTrue(len(network_collection) > 0)

    def checkSerializeDeserialize(self, collection:NetworkCollection):
        serialization_str = collection.serialize()
        self.assertTrue(isinstance(serialization_str, str))
        #
        new_collection = NetworkCollection.deserialize(serialization_str)
        self.assertTrue(isinstance(new_collection, NetworkCollection))
        self.assertTrue(collection == new_collection)
    
    def testSerializeDeserialize1(self):
        if IGNORE_TEST:
            return
        self.checkSerializeDeserialize(self.collection)

    def testSerializeDeserialize2(self):
        if IGNORE_TEST:
            return
        def test(num_network:int=5, array_size:int=5):
            collection = NetworkCollection.makeRandomCollection(num_species=array_size, num_reaction=array_size,
                num_network=num_network)
            self.checkSerializeDeserialize(collection)
        #
        test()
        test(num_network=10, array_size=10)



if __name__ == '__main__':
    unittest.main(failfast=True)