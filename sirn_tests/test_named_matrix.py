from sirn.named_matrix import NamedMatrix  # type: ignore

import copy
import numpy as np
import unittest
import time


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
MAT2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


#############################
# Tests
#############################
class TestNamedMatrix(unittest.TestCase):

    def setUp(self):
        self.array = copy.copy(MAT)
        self.named_matrix = NamedMatrix(MAT.copy(), row_ids=[(1, 0), (0, 1), (0, 0)],
                                        column_ids=['d', 'e', 'f'])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        row_name = "rows"
        column_name = "columns"
        named_matrix = NamedMatrix(MAT2.copy(), row_ids=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_ids=['d', 'e', 'f'],
                                    row_name=row_name, column_name=column_name)
        self.assertTrue(np.all(named_matrix.values == MAT2))
        self.assertTrue("1 0" in str(named_matrix))
        self.assertTrue("e" in str(named_matrix))
        self.assertTrue(row_name in str(named_matrix))
        self.assertTrue(column_name in str(named_matrix))

    def testEq(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.named_matrix == self.named_matrix)
        #
        named_matrix = NamedMatrix(MAT2.copy(), row_ids=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_ids=['d', 'e', 'f'])
        self.assertFalse(self.named_matrix == named_matrix)

    def testLe(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.named_matrix <= self.named_matrix)
        #
        named_matrix = NamedMatrix(MAT2.copy(), row_ids=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_ids=['d', 'e', 'f'])
        self.assertFalse(self.named_matrix <= named_matrix)
        #
        mat = MAT.copy()
        mat[0, 0] = 10
        named_matrix = NamedMatrix(mat, row_ids=[(1, 0), (0, 1), (0, 0)],
                                        column_ids=['d', 'e', 'f'])
        self.assertFalse(named_matrix <= self.named_matrix)

    def testTemplate(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.template()
        self.assertTrue(named_matrix == self.named_matrix)
        #
        mat = MAT.copy()
        mat[0, 0] = 10
        named_matrix = self.named_matrix.template(matrix=mat)
        self.assertFalse(named_matrix == self.named_matrix)
        self.assertTrue(named_matrix.isCompatible(self.named_matrix))

    def testGetSubNamedMatrix(self):
        if IGNORE_TEST:
            return
        result = self.named_matrix.getSubNamedMatrix(row_ids=[(1, 0), (0, 1)], column_ids=['d', 'e'])
        named_matrix = result.named_matrix
        self.assertTrue(np.all(named_matrix.values == np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.all(named_matrix.row_ids == np.array([(1, 0), (0, 1)])))
        self.assertTrue(np.all(named_matrix.column_ids == np.array(['d', 'e'])))

    def testGetSubMatrix(self):
        if IGNORE_TEST:
            return
        matrix = self.named_matrix.getSubMatrix(row_idxs=range(2), column_idxs=range(2))
        self.assertTrue(np.all(matrix.values == np.array([[1, 0], [0, 1]])))

    def testPerformance(self):
        if IGNORE_TEST:
            return
        num_row = 100
        num_column = 100
        mat = np.random.randint(-10, 10, (num_row, num_column))
        named_matrix = NamedMatrix(mat)
        def timeit(num_iteration=1000, is_named_matrix=True):
            t0 = time.time()
            for _ in range(num_iteration):
                if is_named_matrix:
                    _ = named_matrix.getSubNamedMatrix(row_ids=range(10), column_ids=range(10))
                else:
                    _ = named_matrix.getSubMatrix(row_idxs=range(10), column_idxs=range(10))
            t1 = time.time()
            return t1 - t0
        #
        time_named_matrix = timeit(is_named_matrix=True)
        time_matrix = timeit(is_named_matrix=False)
        self.assertTrue(time_named_matrix/time_matrix > 100)
        #print(f"TimeNamed: {time_named_matrix}", f"TimeMatrix: {time_matrix}")

    def testCopyEquals(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.copy()
        self.assertTrue(named_matrix == self.named_matrix)
        

if __name__ == '__main__':
    unittest.main(failfast=True)