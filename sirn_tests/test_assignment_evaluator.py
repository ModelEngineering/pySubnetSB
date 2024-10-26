import sirn.constants as cn  # type: ignore
from src.sirn.assignment_evaluator import AssignmentEvaluator, _Assignment, DEBUG  # type: ignore
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
        self.num_assignment = ROW_ASSIGNMENT_ARR.shape[0]*COLUMN_ASSIGNMENT_ARR.shape[0]
        self.evaluator = AssignmentEvaluator(REFERENCE_ARR, TARGET_ARR)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.evaluator.reference_arr, np.ndarray))

    def testParallelEvaluateBasic(self):
        if IGNORE_TEST:
            return
        assignment_pairs = self.evaluator.parallelEvaluate(ROW_ASSIGNMENT_ARR, COLUMN_ASSIGNMENT_ARR)
        assignment_pairs = self.evaluator.parallelEvaluate(ROW_ASSIGNMENT_ARR, COLUMN_ASSIGNMENT_ARR)
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
            num_assignment = 100
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
            evaluator = AssignmentEvaluator(reference_arr, target_arr, max_batch_size=int(2e6))
            assignment_pairs = evaluator.parallelEvaluate(row_assignment_arr, column_assignment_arr)
            # Check the result
            true_assignment_pair = AssignmentPair(row_assignment=true_row_assignment_arr,
                column_assignment=true_column_assignment_arr)
            self.assertEqual(assignment_pairs[-1], true_assignment_pair)

    def testParallelEvaluateScale1(self):
        if IGNORE_TEST:
            return
        for _ in range(2):
            reference_size = np.random.randint(3, 200)
            reference_network = Network.makeRandomNetworkByReactionType(num_species=reference_size,
                  num_reaction=2*reference_size)
            reference_arr = reference_network.reactant_nmat.values
            _, num_reference_column = reference_arr.shape
            target_network, assignment_pair = reference_network.permute()
            reaction_assignment = np.array([assignment_pair.reaction_assignment,
                np.random.permutation(np.arange(num_reference_column))])
            column_assignment = _Assignment(reaction_assignment)
            spurious_assignment_arr = np.random.permutation(range(len(assignment_pair.species_assignment)))
            species_assignment = np.array([assignment_pair.species_assignment, spurious_assignment_arr])
            row_assignment = _Assignment(species_assignment)
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values, target_network.reactant_nmat.values)
            assignment_pairs = evaluator.parallelEvaluate(row_assignment.array, column_assignment.array)
            assigned_target_arr = target_network.reactant_nmat.values[assignment_pairs[0].row_assignment, :]
            assigned_target_arr = assigned_target_arr[:, assignment_pairs[0].column_assignment]
            self.assertTrue(np.all(reference_network.reactant_nmat.values == assigned_target_arr))
    
    def testParallelEvaluateScaleFill(self):
        if IGNORE_TEST:
            return
        for _ in range(2):
            num_assignment = 100
            num_species_desired = np.random.randint(100, 200)
            num_reaction_desired = num_species_desired + 1
            fill_size = 3*num_species_desired
            reference_network = Network.makeRandomNetworkByReactionType(num_species=num_species_desired,
                  num_reaction=num_reaction_desired)
            num_species, num_reaction = reference_network.reactant_nmat.values.shape
            target_network = reference_network.fill(num_fill_reaction=fill_size, num_fill_species=fill_size,
                  is_permute=False)
            true_species_assignment_arr = np.array(range(num_species))
            true_reaction_assignment_arr = np.array(range(num_reaction))
            true_assignment_pair = AssignmentPair(row_assignment=true_species_assignment_arr,
                column_assignment=true_reaction_assignment_arr)
            reaction_assignment_arr = np.array([np.random.permutation(np.arange(num_reaction)) for _ in range(num_assignment)])
            reaction_assignment_arr[-1] = true_reaction_assignment_arr
            species_assignment_arr = np.array([np.random.permutation(np.arange(num_species)) for _ in range(num_assignment)])
            species_assignment_arr[-1] = true_species_assignment_arr
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values, target_network.reactant_nmat.values)
            assignment_pairs = evaluator.parallelEvaluate(species_assignment_arr, reaction_assignment_arr,
                    total_process=10)
            trues = [assignment_pair == true_assignment_pair for assignment_pair in assignment_pairs]
            self.assertTrue(any(trues))
        
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
            evaluator = AssignmentEvaluator(reference_network.reactant_nmat.values, target_network.reactant_nmat.values)
            # Evaluate the arrays
            candidate_assignment_pairs = [AssignmentPair(row_assignment=xv, column_assignment=yv)
                                for xv, yv in itertools.product(row_assignment.array, column_assignment.array)]
            successful_assignment_pairs = evaluator.evaluateAssignmentPairs(candidate_assignment_pairs)
            assigned_target_arr = target_network.reactant_nmat.values[successful_assignment_pairs[0].row_assignment, :]
            assigned_target_arr = assigned_target_arr[:, candidate_assignment_pairs[0].column_assignment]
            self.assertTrue(np.all(reference_network.reactant_nmat.values == assigned_target_arr))

    def testParallelEvaluateSimple(self):
        if IGNORE_TEST:
            return
        for total_process in [1, 10]:
            results = self.evaluator.parallelEvaluate(ROW_ASSIGNMENT_ARR, COLUMN_ASSIGNMENT_ARR,
                  total_process=total_process)
            result = results[0]
            for pair in [result.species_assignment, result.reaction_assignment]:
                self.assertTrue(np.all(pair == [2, 0]))


if __name__ == '__main__':
    unittest.main(failfast=False)