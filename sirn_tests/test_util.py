from sirn import util  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False


#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def testHashArray(self):
        if IGNORE_TEST:
            return
        arr = np.array([1, 2, 3])
        hash_val = util.hashArray(arr)
        self.assertTrue("int" in str(type(hash_val)))
        self.assertTrue(hash_val > 0)

    def testHashArray2(self):
        # Test distribution of hash values
        if IGNORE_TEST:
            return
        def makeInt(count:int=3)->int:
            return int(1e6*np.random.randint(0, count) + 1e3*np.random.randint(0, count) \
                        + np.random.randint(0, count))
        def makeArray(n: int)->np.array:
            result = [makeInt() for _ in range(n)]
            return np.array(result)
        result_dct = {}
        num_arr = int(1e4)
        for _ in range(num_arr):
            arr = makeArray(5)
            hash_val = util.hashArray(arr)
            if hash_val not in result_dct:
                result_dct[hash_val] = []
            result_dct[hash_val].append(arr)
        lengths = np.array([len(v) for v in result_dct.values()])
        frac_collision = sum(lengths > 1)/num_arr
        self.assertLess(frac_collision, 0.1)

    def testString2Array(self):
        if IGNORE_TEST:
            return
        def test(array):
            array_str = str(array)
            arr = util.string2Array(array_str)
            self.assertTrue(np.all(arr == arr))
        #
        array = np.array(range(10))
        test(array)
        test(np.reshape(array, (2,5)))

    def testHashBug(self):
        if IGNORE_TEST:
            return
        result = util.hashArray(np.array([10200, 10200]))
        self.assertTrue(result > 0)

    def testTimeit(self):
        if IGNORE_TEST:
            return
        util.IS_TIMEIT = IGNORE_TEST
        @util.timeit
        def test():
            time.sleep(1)
        test()
        util.IS_TIMEIT = False


if __name__ == '__main__':
    unittest.main()