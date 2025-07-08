from pySubnetSB.assignment_evaluator_worker import AssignmentEvaluatorWorker  # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore

import copy
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
REFERENCE_ARR = np.array([
    [1, 0],
    [1, 1],
])
TARGET_ARR = np.array([
    [1, 0, 0],
    [1, 1, 1],
    [0, 1, 0],  # Consider an assignment w/o this row
])
VALID_ASSIGNMENT_PAIR = AssignmentPair(row_assignment=[0, 1], column_assignment=[0, 2])
INVALID_ASSIGNMENT_PAIR = AssignmentPair(row_assignment=[0, 1], column_assignment=[0, 1])

#############################
# Tests
#############################
class TestAssignmentEvaluatorWorker(unittest.TestCase):
    """
    Test the constructor of the AssignmentEvaluatorWorker class.
    """
    def setUp(self):
        self.worker = AssignmentEvaluatorWorker(REFERENCE_ARR, TARGET_ARR, 1000)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.worker, AssignmentEvaluatorWorker))
        self.assertTrue(isinstance(self.worker.reference_arr, np.ndarray))
        self.assertTrue(isinstance(self.worker.target_arr, np.ndarray)) 

    def testFilterColumnConstraintForAssignmentPairsValid(self):
        if IGNORE_TEST:
            return
        assignment_pairs = self.worker.filterColumnConstraintForAssignmentPairs([VALID_ASSIGNMENT_PAIR])
        self.assertTrue(np.all(assignment_pairs[0].species_assignment == VALID_ASSIGNMENT_PAIR.species_assignment))
        self.assertTrue(np.all(assignment_pairs[0].reaction_assignment == VALID_ASSIGNMENT_PAIR.reaction_assignment))
    
    def testFilterColumnConstraintForAssignmentPairsInvalid(self):
        if IGNORE_TEST:
            return
        assignment_pairs = self.worker.filterColumnConstraintForAssignmentPairs([INVALID_ASSIGNMENT_PAIR])
        self.assertTrue(len(assignment_pairs) == 0)

if __name__ == '__main__':
    unittest.main()