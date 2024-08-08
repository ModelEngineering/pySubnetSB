from sirn.criteria_vector import CriteriaVector  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[0, 1], [1, 0], [0, 0]])


#############################
# Tests
#############################
class TestCriteriaVector(unittest.TestCase):

    def setUp(self):
        self.criteria_vector = CriteriaVector()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.criteria_vector.criteria_functions), 0)
        self.assertTrue(self.criteria_vector.criteria_functions[0](-1))
        self.assertFalse(self.criteria_vector.criteria_functions[0](2))
        self.assertEqual(len(self.criteria_vector.criteria_strs), len(self.criteria_vector.criteria_functions))


if __name__ == '__main__':
    unittest.main()