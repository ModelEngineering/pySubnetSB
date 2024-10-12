import sirn.constants as cn  # type: ignore
from src.sirn.assignment_evaluator import AssignmentEvaluator, ComparisonCriteria, _Assignment  # type: ignore

import numpy as np
import copy
import unittest


IGNORE_TEST = False
IS_PLOT = False
TARGET_ARR = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
REFERENCE_ARR = np.array([[9, 7], [3, 1]])
TRUE_ROW_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_ROW_ASSIGNMENT_ARR = np.array([0, 2])
TRUE_COLUMN_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_COLUMN_ASSIGNMENT_ARR = np.array([0, 2])
ROW_ASSIGNMEN_ARR = np.array([TRUE_ROW_ASSIGNMENT_ARR, FALSE_ROW_ASSIGNMENT_ARR])
COLUMN_ASSIGNMEN_ARR = np.array([TRUE_COLUMN_ASSIGNMENT_ARR, FALSE_COLUMN_ASSIGNMENT_ARR])


class TestAssignmentEvaluator(unittest.TestCase):

    def setUp(self):
        criteria = ComparisonCriteria(is_equality=True)
        self.num_assignment = ROW_ASSIGNMEN_ARR.shape[0]*COLUMN_ASSIGNMEN_ARR.shape[0]
        self.evaluator = AssignmentEvaluator(REFERENCE_ARR, TARGET_ARR, comparison_criteria=criteria)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.evaluator.comparison_criteria, ComparisonCriteria))

    def testMakeBatch(self):
        if IGNORE_TEST:
            return
        row_assignment = _Assignment(ROW_ASSIGNMEN_ARR)
        column_assignment = _Assignment(COLUMN_ASSIGNMEN_ARR)
        big_reference_arr, big_target_arr = self.evaluator._makeBatch(0, 3, row_assignment, column_assignment)
        # Check if the first assignment is True and the resut are False
        satisfy_idx = np.sum(big_reference_arr == big_target_arr, axis=1) == REFERENCE_ARR.shape[1]
        satisfy_idx = np.reshape(satisfy_idx, (self.num_assignment, self.evaluator.num_reference_column))
        satisfy_idx = np.sum(satisfy_idx, axis=1) == self.evaluator.num_reference_column
        self.assertTrue(satisfy_idx[0])
        self.assertFalse(np.all(satisfy_idx[1:]))


if __name__ == '__main__':
    unittest.main(failfast=False)