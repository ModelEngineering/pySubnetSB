import sirn.constants as cn  # type: ignore
from sirn.network import Network # type: ignore

import numpy as np
import copy
import pandas as pd # type: ignore
import tellurium as te  # type: ignore
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_ITERATION = 10
NETWORK_NAME = "test"
BIG_NETWORK = """
J0: S1 -> S1 + S2; k1*S1;
J1: S2 -> S3; k2*S2;
J2: S3 -> ; k3*S3;
J3: S3 -> S4; k4*S3;
J4: S4 -> S5; k5*S4;

k1 = 1
k2 = 1
k3 = 1
k4 = 1
k5 = 1
S1 = 10
S2 = 0
S3 = 0
S4 = 0
S5 = 0
"""
NETWORK1 = """
J0: S1 -> S1 + S2; k1*S1;
J1: S2 -> S3; k2*S2;

k1 = 1
k2 = 1
S1 = 10
S2 = 0
S3 = 0
"""
NETWORK2 = """
// Strong structural identity with Network1
Ja: S2 -> S2 + S1; k1*S2;
Jb: S1 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK3 = """
// Weak sructural identity with Network1
Ja: 2 S1 -> 2 S1 + S2; k1*S1*S1
Jb: S2 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK4 = """
// Not structurally identical with Network1
Ja: S1 -> S2; k1*S2;
Jb: S1 -> S3; k2*S1;

k1 = 3
k2 = 1
S1 = 1
S2 = 0
S3 = 0
"""
NETWORK = Network.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = copy.deepcopy(NETWORK)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        self.assertTrue("int" in str(type(self.network.weak_hash)))
        self.assertTrue("int" in str(type(self.network.strong_hash)))

    def makeRandomNetwork(self, num_species=5, num_reaction=5):
        big_reactant_mat = np.random.randint(0, 2, (num_species, num_reaction))
        big_product_mat = np.random.randint(0, 2, (num_species, num_reaction))
        return Network(big_reactant_mat, big_product_mat)

    def testIsStructurallyIdenticalBasic(self):
        if IGNORE_TEST:
            return
        def permuteArray(arr, row_perm, column_perm):
            new_arr = arr.copy()
            new_arr = new_arr[row_perm, :]
            new_arr = new_arr[:, column_perm]
            return new_arr
        #
        target, assignment_pair = self.network.permute()
        result = self.network.isStructurallyIdentical(target, identity=cn.ID_WEAK)
        self.assertTrue(np.all(self.network.species_names == target.species_names[assignment_pair.species_assignment]))
        self.assertTrue(np.all(self.network.reaction_names == target.reaction_names[assignment_pair.reaction_assignment]))
        self.assertTrue(result)
        result = self.network.isStructurallyIdentical(self.network, identity=cn.ID_STRONG)
        self.assertTrue(result)

    def testIsStructurallyIdenticalDoubleFail(self):
        if IGNORE_TEST:
            return
        # Double the size of the target. Look for exact match. Should fail.
        reactant_arr = np.vstack([self.network.reactant_nmat.values]*2)
        reactant_arr = np.hstack([reactant_arr]*2)
        product_arr = np.vstack([self.network.product_nmat.values]*2)
        product_arr = np.hstack([product_arr]*2)
        target_network = Network(reactant_arr, product_arr)
        result = self.network.isStructurallyIdentical(target_network, is_subset=False, identity=cn.ID_WEAK)
        self.assertFalse(result)
    
    def testIsStructurallyIdenticalDoubleSubset(self):
        if IGNORE_TEST:
            return
        reactant_arr = np.hstack([self.network.reactant_nmat.values]*2)
        product_arr = np.hstack([self.network.product_nmat.values]*2)
        target_network = Network(reactant_arr, product_arr)
        result = self.network.isStructurallyIdentical(target_network, is_subset=True, identity=cn.ID_WEAK)
        self.assertTrue(result)

    def testIsStructurallyIdenticalSimpleRandomlyPermute(self):
        if IGNORE_TEST:
            return
        target, _ = self.network.permute()
        result = self.network.isStructurallyIdentical(target, identity=cn.ID_WEAK)
        self.assertTrue(result)
        result = self.network.isStructurallyIdentical(target, identity=cn.ID_STRONG)
        self.assertTrue(result)

    def checkEquivalent(self, network1:Network, network2:Network=None, identity:str=cn.ID_STRONG):
        if network2 is None:
            network2 = self.network
        result = network1.isStructurallyIdentical(network2, identity=identity)
        self.assertTrue(result)
        result = network2.isStructurallyIdentical(network1, identity=identity)
        self.assertTrue(result)
    
    def testIsStructurallyIdenticalScaleRandomlyPermuteTrue(self):
        if IGNORE_TEST:
            return
        def test(reference_size, fill_factor=1, num_iteration=2*NUM_ITERATION):
            for identity in [cn.ID_WEAK, cn.ID_STRONG]:
                num_success = 0
                for _ in range(num_iteration):
                    reference = Network.makeRandomNetworkByReactionType(reference_size)
                    target = reference.fill(num_fill_reaction=fill_factor*reference_size,
                        num_fill_species=fill_factor*reference_size)
                    result = reference.isStructurallyIdentical(target, identity=identity, is_subset=True,
                            max_num_assignment=5000)
                    num_success += bool(result)
                    #self.assertTrue(bool(result))
                succ_frac = num_success/num_iteration
                self.assertGreater(succ_frac, 0.5)
                #print(f"Success rate for {identity}: {num_success/num_iteration}")
        #
        for fill_factor in [1, 2]:
            for size in [3, 5, 7]:
                test(size, fill_factor=fill_factor)

    def testIsStructurallyIdenticalScaleRandomlyPermuteFalse(self):
        if IGNORE_TEST:
            return
        def test(reference_size, target_factor=1, num_iteration=NUM_ITERATION):
            for _ in range(num_iteration):
                for identity in [cn.ID_STRONG]:
                    for is_subset in [False]:
                        reference = Network.makeRandomNetworkByReactionType(reference_size)
                        target = Network.makeRandomNetworkByReactionType(target_factor*reference_size)
                        # Analyze
                        result = reference.isStructurallyIdentical(target, identity=identity, is_subset=is_subset)
                        self.assertFalse(result)
        #
        for size in [5, 10, 20, 40]:
            test(size, target_factor=5)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            network = self.makeRandomNetwork(10, 10)
            serialization_str = network.serialize()
            self.assertTrue(isinstance(serialization_str, str))
            new_network = Network.deserialize(serialization_str)
            self.assertEqual(network, new_network)
    
    def testIsIsomorphic(self):
        if IGNORE_TEST:
            return
        def test(reference_size, is_isomorphic=True, num_iteration=NUM_ITERATION):
            for _ in range(num_iteration):
                    reference = Network.makeRandomNetworkByReactionType(reference_size)
                    target, _ = reference.permute()
                    if not is_isomorphic:
                        if target.reactant_nmat.values[0, 0] == 1:
                            target.reactant_nmat.values[0, 0] = 0
                        else:
                            target.reactant_nmat.values[0, 0] = 1
                    target = Network(target.reactant_nmat.values, target.product_nmat.values)
                    result = reference.isIsomorphic(target)
                    self.assertEqual(result, is_isomorphic)
        #
        test(10, is_isomorphic=True)
        test(10, is_isomorphic=False)

    def testCompare(self):
        if IGNORE_TEST:
            return
        target_arr = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reference_arr = np.array([ [9, 7], [3, 1]])
        true_species_assignment_arr = np.array([2, 0])
        false_species_assignment_arr = np.array([0, 1])
        # Index of true species assignment is 0
        species_assignment_arr = np.vstack([true_species_assignment_arr, false_species_assignment_arr])
        true_reaction_assignment_arr = np.array([2, 0])
        false_reaction_assignment_arr = np.array([1, 0])
        # Index of true reaction assignment is 1
        reaction_assignment_arr = np.vstack([false_reaction_assignment_arr, true_reaction_assignment_arr])
        # Index of true result is 1 + 0*num_species = 1
        result_arr = self.network._compare(reference_arr, target_arr, species_assignment_arr, reaction_assignment_arr)
        self.assertTrue(result_arr[1])
        self.assertTrue(not result_arr[i] for i in range(4) if i != 1)

    def testCompareScale(self):
        if IGNORE_TEST:
            return
        target_size = 10
        reference_size = 5
        num_assignment = 4
        #####
        def makeAssignment(arr):
            extended_arr = [arr]
            hashes = [hash(str(arr))]
            while len(extended_arr) < num_assignment:
                new_arr = np.random.permutation(arr)
                new_hash = hash(str(new_arr))
                if new_hash in hashes:
                    continue
                extended_arr.append(new_arr)
                hashes.append(new_hash)
            return np.vstack(extended_arr)
        #####
        target_arr = np.random.randint(0, 10, (target_size, target_size))
        true_species_assignment_arr = np.random.permutation(range(target_size))[:reference_size]
        species_assignment_arr = makeAssignment(true_species_assignment_arr)
        true_reaction_assignment_arr = np.random.permutation(range(target_size))[:reference_size]
        reaction_assignment_arr = makeAssignment(true_reaction_assignment_arr)
        reference_arr = target_arr[true_species_assignment_arr, :]
        reference_arr = reference_arr[:, true_reaction_assignment_arr]
        result_arr = self.network._compare(reference_arr, target_arr, species_assignment_arr, reaction_assignment_arr)
        self.assertTrue(result_arr[0])
        self.assertTrue(not v for v in result_arr[1:])


if __name__ == '__main__':
    unittest.main(failfast=True)