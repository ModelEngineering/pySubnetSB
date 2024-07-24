from sirn.named_matrix import NamedMatrix  # type: ignore

import copy
import numpy as np
import unittest


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
        named_matrix = NamedMatrix(MAT2.copy(), row_ids=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_ids=['d', 'e', 'f'])
        self.assertTrue(np.all(named_matrix.matrix == MAT2))
        self.assertTrue("(1, 0)" in str(named_matrix))
        self.assertFalse("(0, 3)" in str(named_matrix))

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

    def testGetSubMatrix(self):
        if IGNORE_TEST:
            return
        subset_result = self.named_matrix.getSubMatrix(row_ids=[(1, 0), (0, 1)], column_ids=['d', 'e'])
        self.assertTrue(np.all(subset_result.matrix == np.array([[1, 0], [0, 1]])))
        #
        subset_result = self.named_matrix.getSubMatrix(row_ids=[(0, 0), (0, 1)], column_ids=['d', 'e'])
        self.assertTrue(np.all(subset_result.matrix == np.array([[0, 1], [0, 0]])))
        #
        with self.assertRaises(ValueError):
            subset_result = self.named_matrix.getSubMatrix(row_ids=[(1, 1), (0, 1)], column_ids=['d', 'e', 'f'])

    def testGetSubNamedMatrix(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.getSubNamedMatrix(row_ids=[(1, 0), (0, 1)], column_ids=['d', 'e'])
        self.assertTrue(np.all(named_matrix.matrix == np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.all(named_matrix.row_ids == np.array([(1, 0), (0, 1)])))
        self.assertTrue(np.all(named_matrix.column_ids == np.array(['d', 'e'])))
        

if __name__ == '__main__':
    unittest.main()