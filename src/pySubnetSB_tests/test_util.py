import pySubnetSB.util as util # type: ignore

import os
import time
import numpy as np
import unittest
from functools import cmp_to_key


IGNORE_TEST = False
IS_PLOT = False

#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

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
        with self.assertRaises(ValueError):
            test(ndim=3)

    def testHashArrayScale(self):
        if IGNORE_TEST:
            return
        def test(size=20, num_iteration=100):
            for _ in range(num_iteration):
                row_perm = np.random.permutation(size)
                arr1 = np.random.randint(-2, 3, (size, size))
                arr2 = arr1.copy()
                arr2 = arr2[row_perm, :]
                result1 = util.makeRowOrderIndependentHash(arr1)
                result2 = util.makeRowOrderIndependentHash(arr2)
                self.assertTrue(result1 == result2)
        test(num_iteration=10000)
    
    def testHashMarixPermutations(self):
        if IGNORE_TEST:
            return
        def test(size=50, num_iteration=100):
            for _ in range(num_iteration):
                row_perm = np.random.permutation(size)
                arr1 = np.random.randint(-2, 3, (size, 2*size))
                arr2 = arr1.copy()
                arr2 = arr2[row_perm, :]
                result1 = util.hashMatrix(arr1)
                result2 = util.hashMatrix(arr2)
                self.assertTrue(result1 == result2)
        test(num_iteration=10000)

    def testHashMarixDistinct(self):
        if IGNORE_TEST:
            return
        def test(size=10, num_iteration=100):
            num_true = 0
            for _ in range(num_iteration):
                arr1 = np.random.randint(-2, 3, (size, 2*size))
                arr2 = np.random.randint(-2, 3, (size, 2*size))
                result1 = util.hashMatrix(arr1)
                result2 = util.hashMatrix(arr2)
                num_true += result1 == result2
            self.assertLess(num_true/num_iteration, 0.1)
            #print(f"{size}: fraction of true: {num_true/num_iteration}")
        test(size=3, num_iteration=1000)
        test(size=5, num_iteration=1000)
    
    def testIsLessThan(self):
        if IGNORE_TEST:
            return
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        arr3 = np.array([1, 2, 4 , 5])
        self.assertTrue(util.isArrayLessEqual(arr1, arr2))
        self.assertFalse(util.isArrayLessEqual(arr2, arr1))
        self.assertFalse(util.isArrayLessEqual(arr1, arr3))

    def testArrayToStr(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            big_array = np.random.randint(0, 10, 100)
            big_array = np.reshape(big_array, (10, 10))
            context = util.array2Context(big_array)
            other_array = util.string2Array(context)
            self.assertTrue(np.all(big_array == other_array))
    
    def testArrayToStr2(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            big_array = np.random.randint(0, 10, 100)
            context = util.array2Context(big_array)
            other_array = util.string2Array(context)
            self.assertTrue(np.all(big_array == other_array.flatten()))

    def testPruneArray(self):
        if IGNORE_TEST:
            return
        def test(size=20, max_size=10, num_iteration=100):
            for _ in range(num_iteration):
                arr = np.random.randint(0, 3, (size, size))
                pruned_arr, is_pruned = util.pruneArray(arr, max_size)
                self.assertTrue(is_pruned)
                self.assertTrue(pruned_arr.shape[0] <= max_size)
        #
        test(size=5, max_size=2)
        test()

    def testHashMatrixTrue(self):
        if IGNORE_TEST:
            return
        def test(size=10, num_iteration=100):
            for _ in range(num_iteration):
                row_perm = np.random.permutation(size)
                arr1 = np.random.randint(0, 3, (size, size))
                arr2 = arr1.copy()
                arr2 = arr2[row_perm, :]
                result1 = util.makeRowOrderIndependentHash(arr1)
                result2 = util.makeRowOrderIndependentHash(arr2)
                self.assertTrue(result1 == result2)
        test(num_iteration=10000)

    def testHashMatrixFalse(self):
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=100):
            num_collision = 0
            for _ in range(num_iteration):
                arr1 = np.random.randint(0, 3, (size, size))
                arr2 = np.random.randint(0, 3, (size, size))
                result1 = util.makeRowOrderIndependentHash(arr1)
                result2 = util.makeRowOrderIndependentHash(arr2)
                num_collision += result1 == result2
        test(num_iteration=1000)

    def testListOfLists(self):
        if IGNORE_TEST:
            return
        for num_list in range(2, 20):
            lengths = np.random.randint(2, 10, num_list)
            list_of_lists = [np.random.randint(0, length, length) for length in lengths]
            sample_arr = util.sampleListOfLists(list_of_lists, 10)
            for sample in sample_arr:
                self.assertTrue(np.all([s in list_of_lists[i] for i, s in enumerate(sample)]))

    def testPartitionArray(self):
        if IGNORE_TEST:
            return
        arr = np.random.randint(0, 10, 45)
        arr = np.reshape(arr, (15, 3))
        num_partition = 4
        partitions = util.partitionArray(arr, num_partition)
        self.assertEqual(len(partitions), num_partition)
        lengths = [len(partition) for partition in partitions]
        self.assertTrue(np.max(lengths) - np.min(lengths) <= 1)
        for row in arr:
            is_true = False
            for partition in partitions:
                for partition_row in partition:
                    is_true = is_true or np.all(row == partition_row)
            self.assertTrue(is_true)
    
    def testPartitionArray2(self):
        if IGNORE_TEST:
            return
        arr = [ np.array([1, 0])]
        num_partition = 2
        partitions = util.partitionArray(arr, num_partition)
        self.assertEqual(len(partitions), num_partition-1)

    def testEncodeIntPairDecodeIntPair(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            int1, int2 = (np.random.randint(1000, 100000), np.random.randint(1000, 100000))
            encoded_int = util.encodeIntPair(int1, int2)
            decoded_int1, decoded_int2 = util.decodeIntPair(encoded_int)
            self.assertEqual(int1, decoded_int1)
            self.assertEqual(int2, decoded_int2)
    
    def testEncodeIntPairDecodeIntPairArray(self):
        if IGNORE_TEST:
            return
        size = 10
        for _ in range(100):
            int1 = np.random.randint(1000, 100000, size)
            int2 = np.random.randint(1000, 100000, size)
            encoded_int = util.encodeIntPair(int1, int2)
            decoded_int1, decoded_int2 = util.decodeIntPair(encoded_int)
            self.assertTrue(np.all(int1 == decoded_int1))
            self.assertTrue(np.all(int2 == decoded_int2))

    def testGetAllSubsets(self):
        if IGNORE_TEST:
            return
        arr = [1, 2, 3, 4]
        subsets = util.getAllSubsets(arr)
        self.assertEqual(len(subsets), 2**len(arr))

    def testMakeLocalFileFromURL(self):
        if IGNORE_TEST:
            return
        url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt"
        local_file = util.makeLocalFileFromURL(url)
        self.assertTrue(os.path.isfile(local_file))
        with open(local_file, "r") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1)

    def testGetStringsFromURL(self):
        if IGNORE_TEST:
            return
        url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt"
        strings = util.getStringsFromURL(url)
        self.assertGreater(len(strings), 1)


if __name__ == '__main__':
    unittest.main()