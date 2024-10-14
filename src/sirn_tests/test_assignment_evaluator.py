import sirn.constants as cn  # type: ignore
from src.sirn.assignment_evaluator import AssignmentEvaluator, ComparisonCriteria, _Assignment, DEBUG  # type: ignore
from src.sirn.network import Network # type: ignore
from src.sirn.assignment_pair import AssignmentPair # type: ignore

import itertools
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


class TestAssignmentEvaluator(unittest.TestCase):

    def setUp(self):
        self.criteria = ComparisonCriteria(is_equality=True)
        self.num_assignment = ROW_ASSIGNMENT_ARR.shape[0]*COLUMN_ASSIGNMENT_ARR.shape[0]
        self.evaluator = AssignmentEvaluator(REFERENCE_ARR, TARGET_ARR, comparison_criteria=self.criteria)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.evaluator.comparison_criteria, ComparisonCriteria))

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
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values,
                target_network.reactant_nmat.values, comparison_criteria=self.criteria)
            num_assignment = row_assignment.num_row*column_assignment.num_row
            big_reference_arr, big_target_arr = evaluator._makeBatch(0, num_assignment-1, row_assignment, column_assignment)
            satisfy_comparison_arr = evaluator._compare(big_reference_arr, big_target_arr)
            self.assertTrue(satisfy_comparison_arr[0])
            self.assertFalse(np.all(satisfy_comparison_arr[1:]))

    def testEvaluateAssignmentArrays(self):
        if IGNORE_TEST:
            return
        assignment_pairs = self.evaluator.evaluateAssignmentArrays(ROW_ASSIGNMENT_ARR, COLUMN_ASSIGNMENT_ARR)
        assigned_target_arr = TARGET_ARR[assignment_pairs[0].row_assignment, :]
        assigned_target_arr = assigned_target_arr[:, assignment_pairs[0].column_assignment]
        self.assertTrue(np.all(REFERENCE_ARR == assigned_target_arr))
    
    def testEvaluateAssignmentArraysMultipleBatches(self):
        if IGNORE_TEST:
            return
        for _ in range(2):
            num_row = 100
            num_column = 10
            expansion_factor = 2
            num_assignment = 1000  # Requires a total of 4GB of memory
            # Construct the reference and target arrays
            reference_arr = np.random.randint(0, 10, (num_row, num_column))
            target_arr = np.vstack([np.random.permutation(reference_arr) for _ in range(expansion_factor)])
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
            evaluator = AssignmentEvaluator(reference_arr, target_arr, comparison_criteria=self.criteria,
                max_batch_size=int(2e6))
            assignment_pairs = evaluator.evaluateAssignmentArrays(row_assignment_arr, column_assignment_arr)
            # Check the result
            true_assignment_pair = AssignmentPair(row_assignment=true_row_assignment_arr,
                column_assignment=true_column_assignment_arr)
            if len(assignment_pairs) == 0:
                import pdb; pdb.set_trace()
            self.assertEqual(assignment_pairs[-1], true_assignment_pair)

    def testEvaluateAssignmentArraysScale(self):
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
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values,
                target_network.reactant_nmat.values, comparison_criteria=self.criteria)
            assignment_pairs = evaluator.evaluateAssignmentArrays(row_assignment.array, column_assignment.array)
            assigned_target_arr = target_network.reactant_nmat.values[assignment_pairs[0].row_assignment, :]
            assigned_target_arr = assigned_target_arr[:, assignment_pairs[0].column_assignment]
            self.assertTrue(np.all(reference_network.reactant_nmat.values == assigned_target_arr))
        
    def testEvaluateAssignmentPairs(self):
        if IGNORE_TEST:
            return
        for _ in range(NUM_ITERATION):
            reference_size = np.random.randint(3, 200)
            reference_network = Network.makeRandomNetworkByReactionType(num_species=reference_size,
                  num_reaction=2*reference_size)
            reference_arr = reference_network.reactant_nmat.values
            num_reference_row, num_reference_column = reference_arr.shape
            target_network, true_assignment_pair = reference_network.permute()
            reaction_assignment = np.array([true_assignment_pair.reaction_assignment,
                np.random.permutation(np.arange(num_reference_column))])
            column_assignment = _Assignment(reaction_assignment)
            spurious_assignment_arr = np.random.permutation(range(len(true_assignment_pair.species_assignment)))
            species_assignment = np.array([true_assignment_pair.species_assignment, spurious_assignment_arr])
            row_assignment = _Assignment(species_assignment)
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values,
                target_network.reactant_nmat.values, comparison_criteria=self.criteria)
            # Evaluate the arrays
            candidate_assignment_pairs = [AssignmentPair(row_assignment=xv, column_assignment=yv)
                                for xv, yv in itertools.product(row_assignment.array, column_assignment.array)]
            successful_assignment_pairs = evaluator.evaluateAssignmentPairs(candidate_assignment_pairs)
            assigned_target_arr = target_network.reactant_nmat.values[successful_assignment_pairs[0].row_assignment, :]
            assigned_target_arr = assigned_target_arr[:, candidate_assignment_pairs[0].column_assignment]
            self.assertTrue(np.all(reference_network.reactant_nmat.values == assigned_target_arr))        


if __name__ == '__main__':
    unittest.main(failfast=False)