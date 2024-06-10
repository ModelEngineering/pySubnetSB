from sirn.array_collection import ArrayCollection # type: ignore
from sirn.matrix import Matrix # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])


#############################
# Tests
#############################
class TestMatrixClassifier(unittest.TestCase):

    def setUp(self):
        self.collection = ArrayCollection(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.collection.collection == MAT))
        self.assertTrue(isinstance(self.collection, ArrayCollection))
        self.assertGreater(self.collection.encoding[-1], self.collection.encoding[0])

    def testClassifyArray(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.allclose(self.collection.encoding, np.array([1002000, 2001000])))

    def testEncode2(self):
        # Test random sequences
        if IGNORE_TEST:
            return
        def test(num_iteration=10, size=10, prob0=1/3):
            # Makes a trinary matrix and checks the encodings
            for _ in range(num_iteration):
                matrix = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                arr = matrix[1,:]
                counts = []
                counts.append(np.sum(arr < 0))
                counts.append(np.sum(arr == 0))
                counts.append(np.sum(arr > 0))
                collection = ArrayCollection(matrix)
                new_encoding = 0
                for idx in range(3):
                    new_encoding += counts[idx]*1000**idx
                self.assertTrue(any([e == new_encoding for e in collection.encoding]))
        #
        test()
        test(prob0=2/3)
        test(prob0=1/5)
        

if __name__ == '__main__':
    unittest.main()