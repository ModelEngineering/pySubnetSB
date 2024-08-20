from sirn import constants as cn  # type: ignore
from sirn.pmatrix import PMatrix # type: ignore
from sirn.pmatrix_collection import PMatrixCollection # type: ignore
from sirn.pmc_serializer import PMCSerializer # type: ignore

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
COLLECTION_SIZE = 10
PMATRICES = [PMatrix(np.random.randint(0, 2, (3, 3)), ['r1', 'r2', 'r3'],
     ['s1', 's2', 's3']) for _ in range(COLLECTION_SIZE)]
SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, "Oscillators_DOE_JUNE_10.csv")


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
        self.assertTrue(isinstance(str(self.collection), str))

    def testMakeRandomCollection(self):
        if IGNORE_TEST:
            return
        size = 10
        collection = PMatrixCollection.makeRandomCollection(num_pmatrix=size)
        self.assertTrue(len(collection) == size)

    def makePermutablyIdenticalCollection(self, num_pmatrix:int=5, num_row:int=3, num_column:int=3):
        pmatrix = PMatrix(PMatrix.makeTrinaryMatrix(num_row, num_column))
        pmatrices = [PMatrix(pmatrix.randomize().array) for _ in range(num_pmatrix)]
        return PMatrixCollection(pmatrices)
    
    def testAdd(self):
        if IGNORE_TEST:
            return
        collection1 = self.makePermutablyIdenticalCollection()
        collection2 = self.makePermutablyIdenticalCollection()
        collection = collection1 + collection2
        self.assertTrue(len(collection) == len(collection1) + len(collection2))

    def testCluster(self):
        if IGNORE_TEST:
            return
        # Construct a collection of two sets of permutably identical matrices
        def test(num_collection=2, num_pmatrix=5, num_row=5, num_column=5):
            # Make pm_collection of permutably identical matrices
            pmatrix_collections = [self.makePermutablyIdenticalCollection(
                num_row=num_row, num_column=num_column, num_pmatrix=num_pmatrix)
                for _ in range(num_collection)]
            # Construct the pm_collection to analyze and is the combination of the other pm_collections
            pmatrix_collection = pmatrix_collections[0]
            for i in range(1, num_collection):
                pmatrix_collection += pmatrix_collections[i]
            pmatrix_collections = pmatrix_collection.cluster()
            self.assertEqual(len(pmatrix_collections), num_collection)
            for pmatrix_collection in pmatrix_collections:
                self.assertTrue(str(pmatrix_collection) in str(pmatrix_collections))
        #
        test(num_collection=5, num_pmatrix=1000)
        test()
        test(num_collection=5)
    
    def testCluster1(self):
        if IGNORE_TEST:
            return
        df = pd.read_csv(SERIALIZATION_PATH)
        pmatrix_collection = PMCSerializer.deserialize(df)
        pmatrix_identity_collections = pmatrix_collection.cluster()
        self.assertTrue(len(pmatrix_identity_collections) == len(pmatrix_collection))


if __name__ == '__main__':
    unittest.main()