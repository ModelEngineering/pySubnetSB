import sirn.util as util # type: ignore

import time
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False

#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

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

    def testTimeit(self):
        if IGNORE_TEST:
            return
        util.IS_TIMEIT = IGNORE_TEST
        @util.timeit
        def test():
            time.sleep(1)
        test()
        util.IS_TIMEIT = False

    def testRepeatRow(self):
        if IGNORE_TEST:
            return
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        num_repeat = 2
        result = util.repeatRow(arr, num_repeat)
        for idx in range(arr.shape[0]):
            result_idx1 = num_repeat*idx
            result_idx2 = num_repeat*idx + 1
            self.assertTrue(np.all(result[result_idx1] == arr[idx]))
            self.assertTrue(np.all(result[result_idx2] == arr[idx]))

    def testRepeatArray(self):
        if IGNORE_TEST:
            return
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        num_repeat = 2
        num_row = arr.shape[0]
        result = util.repeatArray(arr, num_repeat)
        for idx in range(num_repeat):
            rows = range(idx*num_row, (idx+1)*num_row)
            self.assertTrue(np.all(result[rows] == arr))

    def testHashArray(self):
        if IGNORE_TEST:
            return
        def test(size=20, ndim=2, num_iteration=100):
            for _ in range(num_iteration):
                row_perm = np.random.permutation(size)
                mat_perm = np.random.permutation(size)
                if ndim == 1:
                    arr1 = np.random.randint(-2, 3, size)
                    arr2 = arr1.copy()
                if ndim == 2:
                    arr1 = np.random.randint(-2, 3, (size, size))
                    arr2 = arr1.copy()
                    arr2 = arr2[row_perm, :]
                if ndim == 3:
                    arr1 = np.random.randint(-2, 3, (size, size, size))
                    arr2 = arr1.copy()
                    arr2 = arr2[mat_perm, :, :]
                    arr2 = arr2[:, row_perm, :]
                result1 = util.makeRowOrderIndependentHash(arr1)
                result2 = util.makeRowOrderIndependentHash(arr2)
                self.assertTrue(result1 == result2)
        #
        test(ndim=1)
        test(ndim=2)
        test(ndim=3)


if __name__ == '__main__':
    unittest.main()