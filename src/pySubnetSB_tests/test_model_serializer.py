from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.network_collection import NetworkCollection  # type: ignore
import pySubnetSB.constants as cn   # type: ignore

import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
MODEL_DIRECTORY = "oscillators"
SERIALIZATION_FILE = os.path.join(cn.DATA_DIR, 'oscillators_serializers.txt')


#############################
# Tests
#############################
class TestModelSerializer(unittest.TestCase):

    def setUp(self):
        self.model_serializer = ModelSerializer.makeOscillatorSerializer(MODEL_DIRECTORY,
              parent_directory=cn.TEST_DIR)
        self.remove()

    def tearDown(self):
        self.remove()

    def remove(self):
        if os.path.exists(SERIALIZATION_FILE):
            os.remove(SERIALIZATION_FILE)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.model_serializer.model_directory.split("/")[-1] == MODEL_DIRECTORY)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        self.model_serializer.serialize()
        model_serializer = ModelSerializer.makeOscillatorSerializer(MODEL_DIRECTORY, parent_directory=cn.TEST_DIR)
        network_collection = model_serializer.deserialize()
        self.assertTrue(isinstance(network_collection, NetworkCollection))
        ffiles = [f for f in os.listdir(os.path.join(cn.TEST_DIR, MODEL_DIRECTORY))
              if not f.endswith('txt')]
        self.assertTrue(len(ffiles) - len(network_collection) <= 1)

    def testDeserializeWithNames(self):
        if IGNORE_TEST:
            return
        SIZE = 5
        self.model_serializer.serialize()
        network_collection = self.model_serializer.deserialize()
        names = [n.network_name for n in network_collection.networks[0:SIZE]]
        new_network_collection = self.model_serializer.deserialize(model_names=names)
        self.assertEqual(len(new_network_collection), SIZE)
        for idx, name in enumerate(names):
            self.assertEqual(name, new_network_collection.networks[idx].network_name)

        

if __name__ == '__main__':
    unittest.main()