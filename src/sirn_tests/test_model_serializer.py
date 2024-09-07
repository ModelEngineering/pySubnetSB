from sirn.model_serializer import ModelSerializer  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore
import sirn.constants as cn   # type: ignore

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
        self.model_serializer = ModelSerializer(MODEL_DIRECTORY, model_parent_dir=cn.TEST_DIR)
        self.remove()

    def tearDown(self):
        self.remove()

    def remove(self):
        if os.path.exists(SERIALIZATION_FILE):
            os.remove(SERIALIZATION_FILE)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.model_serializer.model_directory == MODEL_DIRECTORY)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        self.model_serializer.serialize()
        model_serializer = ModelSerializer(MODEL_DIRECTORY, model_parent_dir=cn.TEST_DIR)
        network_collection = model_serializer.deserialize()
        self.assertTrue(isinstance(network_collection, NetworkCollection))
        ffiles = os.listdir(os.path.join(cn.TEST_DIR, MODEL_DIRECTORY))
        self.assertTrue(len(network_collection) == len(ffiles))

        

if __name__ == '__main__':
    unittest.main()