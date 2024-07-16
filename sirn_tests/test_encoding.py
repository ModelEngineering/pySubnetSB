from sirn import constants as cn  # type: ignore
from sirn.encoding import Encoding  # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
ARRAY = np.array([1, 2, -1, 0, 1])


#############################
# Tests
#############################
class TestEncoding(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.encoding = Encoding(ARRAY)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        sorted_arr = np.sort(ARRAY)
        self.assertTrue(np.allclose(self.encoding.array, ARRAY))
        self.assertTrue(np.allclose(self.encoding.sorted_arr, sorted_arr))
        self.assertTrue(self.encoding.encoding_val == '-1,0,1,1,2')

    def testEqual(self):
        if IGNORE_TEST:
            return
        encoding2 = Encoding(ARRAY)
        self.assertTrue(self.encoding == encoding2)
        #
        array = ARRAY.copy()
        array[0] = 0
        encoding2 = Encoding(array)
        self.assertFalse(self.encoding == encoding2)

    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue("," in str(self.encoding))

    def testIsCompatibleSubset(self):
        if IGNORE_TEST:
            return
        array = np.concatenate([ARRAY, [1, 2, 3]])
        array = np.random.permutation(array)
        encoding2 = Encoding(array)
        self.assertTrue(self.encoding.isCompatibleSubset(encoding2))
        self.assertFalse(encoding2.isCompatibleSubset(self.encoding))




if __name__ == '__main__':
    unittest.main()