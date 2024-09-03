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

    def testFunctions(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.criteria_vector.criteria_functions[0](-1))
        self.assertFalse(self.criteria_vector.criteria_functions[0](2))
        self.assertTrue(self.criteria_vector.criteria_functions[1](0))
        self.assertFalse(self.criteria_vector.criteria_functions[1](2))
        self.assertTrue(self.criteria_vector.criteria_functions[2](1))
        self.assertFalse(self.criteria_vector.criteria_functions[2](2))
        self.assertTrue(self.criteria_vector.criteria_functions[3](2))
        self.assertFalse(self.criteria_vector.criteria_functions[3](0))

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        string = self.criteria_vector.serialize()
        criteria_vector = CriteriaVector.deserialize(string)
        self.assertEqual(self.criteria_vector, criteria_vector)


if __name__ == '__main__':
    unittest.main()