import sirn.constants as cn  # type: ignore
from src.sirn.assignment_evaluator import AssignmentEvaluator, ComparisonCriteria, _Assignment, DEBUG  # type: ignore
from src.sirn.network import Network # type: ignore

import numpy as np
import copy
import unittest


IGNORE_TEST = True
IS_PLOT = False
if IGNORE_TEST:
    DEBUG = True # Use debug mode in the module
TARGET_ARR = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
REFERENCE_ARR = np.array([[9, 7], [3, 1]])
TRUE_ROW_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_ROW_ASSIGNMENT_ARR = np.array([0, 2])
TRUE_COLUMN_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_COLUMN_ASSIGNMENT_ARR = np.array([0, 2])
ROW_ASSIGNMEN_ARR = np.array([TRUE_ROW_ASSIGNMENT_ARR, FALSE_ROW_ASSIGNMENT_ARR])
COLUMN_ASSIGNMEN_ARR = np.array([TRUE_COLUMN_ASSIGNMENT_ARR, FALSE_COLUMN_ASSIGNMENT_ARR])
NUM_ITERATION = 100


class TestAssignmentEvaluator(unittest.TestCase):

    def setUp(self):
        self.criteria = ComparisonCriteria(is_equality=True)
        self.num_assignment = ROW_ASSIGNMEN_ARR.shape[0]*COLUMN_ASSIGNMEN_ARR.shape[0]
        self.evaluator = AssignmentEvaluator(REFERENCE_ARR, TARGET_ARR, comparison_criteria=self.criteria)

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

    def testMakeBatchScale(self):
        #if IGNORE_TEST:
        #    return
        for _ in range(NUM_ITERATION):
            reference_size = np.random.randint(3, 200)
            reference_network = Network.makeRandomNetworkByReactionType(num_species=reference_size,
                  num_reaction=2*reference_size)
            reference_arr = reference_network.reactant_nmat.values
            num_reference_row, num_reference_column = reference_arr.shape
            target_network, assignment_pair = reference_network.permute()
            reaction_assignment = np.array([assignment_pair.reaction_assignment,
                np.random.permutation(np.arange(num_reference_column))])
            column_assignment = _Assignment(reaction_assignment)
            spurious_assignment_arr = np.random.permutation(range(len(assignment_pair.species_assignment)))
            species_assignment = np.array([assignment_pair.species_assignment, spurious_assignment_arr])
            row_assignment = _Assignment(species_assignment)
            num_assignment = row_assignment.num_row*column_assignment.num_row
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values,
                target_network.reactant_nmat.values, comparison_criteria=self.criteria)
            num_assignment = row_assignment.num_row*column_assignment.num_row
            big_reference_arr, big_target_arr = evaluator._makeBatch(0, num_assignment-1, row_assignment, column_assignment)
            flattened_satisfy_column_arr = np.sum(big_reference_arr == big_target_arr, axis=1) == num_reference_column
            satisfy_row_arr = np.reshape(flattened_satisfy_column_arr, (num_assignment, num_reference_row))
            satisfy_comparison_arr = np.sum(satisfy_row_arr, axis=1) == num_reference_row
            self.assertTrue(satisfy_comparison_arr[0])
            self.assertFalse(np.all(satisfy_comparison_arr[1:]))


if __name__ == '__main__':
    unittest.main(failfast=False)