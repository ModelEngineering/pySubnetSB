import sirn.constants as cn  # type: ignore
from src.sirn.assignment_evaluator_worker import AssignmentEvaluatorWorker, _Assignment  # type: ignore
from src.sirn.network import Network # type: ignore
from src.sirn.assignment_pair import AssignmentPair # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
if IGNORE_TEST:
    DEBUG = True # Use debug mode in the module
TARGET_ARR = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
REFERENCE_ARR = np.array([[9, 7], [3, 1]])
TRUE_ROW_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_ROW_ASSIGNMENT_ARR = np.array([0, 2])
TRUE_COLUMN_ASSIGNMENT_ARR = np.array([2, 0])
FALSE_COLUMN_ASSIGNMENT_ARR = np.array([0, 2])
ROW_ASSIGNMENT_ARR = np.array([TRUE_ROW_ASSIGNMENT_ARR, FALSE_ROW_ASSIGNMENT_ARR])
COLUMN_ASSIGNMENT_ARR = np.array([TRUE_COLUMN_ASSIGNMENT_ARR, FALSE_COLUMN_ASSIGNMENT_ARR])
NUM_ITERATION = 100
MAX_BATCH_SIZE = 1000


class TestAssignmentEvaluator(unittest.TestCase):

    def setUp(self):
        self.num_assignment = ROW_ASSIGNMENT_ARR.shape[0]*COLUMN_ASSIGNMENT_ARR.shape[0]
        self.evaluator = AssignmentEvaluatorWorker(REFERENCE_ARR, TARGET_ARR, max_batch_size=MAX_BATCH_SIZE)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.evaluator.reference_arr, np.ndarray))

    def testMakeBatch(self):
        if IGNORE_TEST:
            return
        row_assignment = _Assignment(ROW_ASSIGNMENT_ARR)
        column_assignment = _Assignment(COLUMN_ASSIGNMENT_ARR)
        big_reference_arr, big_target_arr = self.evaluator._makeBatch(0, 3, row_assignment, column_assignment)
        # Check if the first assignment is True and the resut are False
        satisfy_idx = np.sum(big_reference_arr == big_target_arr, axis=1) == REFERENCE_ARR.shape[1]
        satisfy_idx = np.reshape(satisfy_idx, (self.num_assignment, self.evaluator.num_reference_column))
        satisfy_idx = np.sum(satisfy_idx, axis=1) == self.evaluator.num_reference_column
        self.assertTrue(satisfy_idx[0])
        self.assertFalse(np.all(satisfy_idx[1:]))

    def testCompareMakeBatchScale(self):
        if IGNORE_TEST:
            return
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
            evaluator = AssignmentEvaluatorWorker(reference_network.reactant_nmat.values,
                target_network.reactant_nmat.values, max_batch_size=MAX_BATCH_SIZE)
            num_assignment = row_assignment.num_row*column_assignment.num_row
            big_reference_arr, big_target_arr = evaluator._makeBatch(0, num_assignment-1, row_assignment, column_assignment)
            for is_close in [False, True]:
                satisfy_comparison_arr = evaluator.compare(big_reference_arr, big_target_arr, evaluator.num_reference_row,
                      is_close)
                self.assertTrue(satisfy_comparison_arr[0])
                self.assertFalse(np.all(satisfy_comparison_arr[1:]))

    def testEvaluateAssignmentArrays(self):
        if IGNORE_TEST:
            return
        process_num, total_process = 0, 1
        assignment_pairs = self.evaluator.evaluateAssignmentArrays(process_num, total_process,
              ROW_ASSIGNMENT_ARR, COLUMN_ASSIGNMENT_ARR, is_report=False)
        assigned_target_arr = TARGET_ARR[assignment_pairs[0].row_assignment, :]
        assigned_target_arr = assigned_target_arr[:, assignment_pairs[0].column_assignment]
        self.assertTrue(np.all(REFERENCE_ARR == assigned_target_arr))
    
    def testEvaluateAssignmentArraysMultipleBatches(self):
        if IGNORE_TEST:
            return
        for _ in range(5):
            num_row = 20
            num_column = 10
            expansion_factor = 2
            num_assignment = 100  # Requires a total of 4GB of memory
            # Construct the reference and target arrays
            reference_arr = np.random.randint(0, 10, (num_row, num_column))
            target_arr = np.vstack([np.random.randint(0, 10, (num_row, num_column)) for _ in range(expansion_factor)])
            target_arr = np.vstack([target_arr, reference_arr])
            # Construct the assignment arrays
            true_row_assignment_arr = np.array(range(expansion_factor*num_row, (expansion_factor+1)*num_row))
            row_assignment_arr = np.array([np.random.permutation(range(num_row)) for _ in range(num_assignment)])
            row_assignment_arr[-1] = true_row_assignment_arr
            true_column_assignment_arr = np.array(range(num_column))
            column_assignment_arr = np.array([np.random.permutation(range(num_column)) for _ in range(num_assignment)])
            # Put the true assignment in a random position
            pos = np.random.randint(0, num_assignment)
            column_assignment_arr[pos] = true_column_assignment_arr
            # Do the evaluation
            evaluator = AssignmentEvaluatorWorker(reference_arr, target_arr, max_batch_size=MAX_BATCH_SIZE)
            process_num, total_process = 0, 1
            assignment_pairs = evaluator.evaluateAssignmentArrays(process_num, total_process, row_assignment_arr, column_assignment_arr)
            # Check the result
            true_assignment_pair = AssignmentPair(row_assignment=true_row_assignment_arr,
                column_assignment=true_column_assignment_arr)
            trues = [assignment_pair == true_assignment_pair for assignment_pair in assignment_pairs]
            self.assertTrue(any(trues))

    def testEvaluateAssignmentArrayScale(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            reference_size = np.random.randint(3, 200)
            reference_network = Network.makeRandomNetworkByReactionType(num_species=reference_size,
                  num_reaction=2*reference_size)
            reference_arr = reference_network.reactant_nmat.values
            _, num_reference_column = reference_arr.shape
            target_network, assignment_pair = reference_network.permute()
            reaction_assignment_arr = np.array([assignment_pair.reaction_assignment,
                np.random.permutation(np.arange(num_reference_column))])
            spurious_assignment_arr = np.random.permutation(range(len(assignment_pair.species_assignment)))
            species_assignment_arr = np.array([assignment_pair.species_assignment, spurious_assignment_arr])
            evaluator = AssignmentEvaluatorWorker(reference_network.reactant_nmat.values,
                  target_network.reactant_nmat.values, max_batch_size=MAX_BATCH_SIZE)
            process_num, total_process = 0, 1
            assignment_pairs = evaluator.evaluateAssignmentArrays(process_num, total_process,
                  species_assignment_arr, reaction_assignment_arr)
            assigned_target_arr = target_network.reactant_nmat.values[assignment_pairs[0].row_assignment, :]
            assigned_target_arr = assigned_target_arr[:, assignment_pairs[0].column_assignment]
            self.assertTrue(np.all(reference_network.reactant_nmat.values == assigned_target_arr))



if __name__ == '__main__':
    unittest.main(failfast=False)