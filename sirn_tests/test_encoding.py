from sirn.encoding import Encoding  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])


#############################
# Tests
#############################
class TestEncoding(unittest.TestCase):

    def setUp(self):
        self.encoding = Encoding(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.encoding.collection == MAT))
        self.assertTrue(isinstance(self.encoding, Encoding))
        self.assertTrue(np.all(self.encoding.encoding_mat[:, 3] == np.array([2, 2, 1])))
        self.assertEqual(len(self.encoding.encoding_dct), 2)

    def testTimingMakeEncodingMat(self):
        if IGNORE_TEST:
            return
        start = time.time()
        count = 0
        for _ in range(1000):
            count += 1
            mat = np.random.randint(0, 2, (100, 100))
            _ = Encoding(mat)
        end = time.time()
        self.assertLess(end - start, 5)

    def testEq(self):
        if IGNORE_TEST:
            return
        encoding = Encoding(MAT)
        self.assertTrue(encoding == self.encoding)
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 2]])
        encoding = Encoding(mat)
        self.assertFalse(encoding == self.encoding)
    
    def testTimingEq(self):
        if IGNORE_TEST:
            return
        mat = np.random.randint(0, 2, (100, 100))
        encoding1 = Encoding(mat)
        encoding2 = Encoding(mat)
        start = time.time()
        for _ in range(10000):
            _ = encoding1 == encoding2
        end = time.time()
        self.assertLess(end - start, 1)


if __name__ == '__main__':
    unittest.main()