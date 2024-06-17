from sirn.pmatrix import PMatrix # type: ignore
from sirn.pmatrix_collection import PMatrixCollection # type: ignore

import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MODEL_NAME = 'model_name'
COLLECTION_SIZE = 10
PMATRICES = [PMatrix(np.random.randint(0, 2, (3, 3)), ['r1', 'r2', 'r3'],

     ['s1', 's2', 's3'], model_name=MODEL_NAME) for _ in range(COLLECTION_SIZE)]


#############################
# Tests
#############################
class TestPMatrixCollection(unittest.TestCase):

    def setUp(self):
        self.collection = PMatrixCollection(PMATRICES)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(len(self.collection) == COLLECTION_SIZE)

    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue(MODEL_NAME in str(self.collection))

    def testMakeRandomCollection(self):
        if IGNORE_TEST:
            return
        size = 10
        collection = PMatrixCollection.makeRandomCollection(num_pmatrix=size)
        self.assertTrue(len(collection) == size)


if __name__ == '__main__':
    unittest.main()