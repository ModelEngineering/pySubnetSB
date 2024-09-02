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
        self.assertTrue("int" in str(type(self.network._weak_hash)))
        self.assertTrue("int" in str(type(self.network._strong_hash)))

    def testMakeCompatibilitySetVector(self):
        if IGNORE_TEST:
            return
        result = self.network.makeCompatibilitySetVector(self.network,
                     cn.OR_SPECIES, identity=cn.ID_WEAK, is_subsets=False)
        for idx, compatibility_set in enumerate(result):
            self.assertTrue(len(compatibility_set) == 1)
            self.assertTrue(compatibility_set[0] == idx)

    def testMakeCompatibilitySetVectorScaleTrue(self):
        if IGNORE_TEST:
            return
        def test(factor=2, num_iteration=100):
            # factor: factor by which the target is scaled
            big_reactant_mat = np.concatenate([self.network.reactant_mat.values for _ in range(factor)])
            big_product_mat = np.concatenate([self.network.product_mat.values for _ in range(factor)])
            big_network = Network(big_reactant_mat, big_product_mat)
            for _ in range(num_iteration):
                result = self.network.makeCompatibilitySetVector(big_network,
                            cn.OR_SPECIES, identity=cn.ID_WEAK, is_subsets=True)
                num_species = self.network.num_species
                for idx, compatibility_set in enumerate(result):
                    self.assertTrue(len(compatibility_set) == factor)
                    for idx2 in compatibility_set:
                        self.assertTrue(np.mod(idx2, num_species) == idx)
        #
        test(2)
        test(20, num_iteration=1000)

    def makeRandomNetwork(self, num_species=5, num_reaction=5):
        big_reactant_mat = np.random.randint(0, 2, (num_species, num_reaction))
        big_product_mat = np.random.randint(0, 2, (num_species, num_reaction))
        return Network(big_reactant_mat, big_product_mat)
    
    def testMakeCompatibilitySetVectorScaleFalse(self):
        if IGNORE_TEST:
            return
        def test(reference_size=10, target_size=20, num_iteration=100):
            for _ in range(num_iteration):
                reference_network = self.makeRandomNetwork(reference_size, reference_size)
                target_network = self.makeRandomNetwork(target_size, target_size)
                result = reference_network.makeCompatibilitySetVector(target_network,
                            cn.OR_SPECIES, identity=cn.ID_WEAK, is_subsets=True)
                size_arr = np.array([len(s) for s in result])
                self.assertTrue(np.all(size_arr <= target_size))

        #
        test(reference_size=10, target_size=20)
        test(reference_size=18, target_size=20)

    def testMakeCompatibleAssignments(self):
        if IGNORE_TEST:
            return
        def test(identity):
            reference_size =3
            target_size = 10
            start_time = time.time()
            for _ in range(10): 
                reference_network = self.makeRandomNetwork(reference_size, reference_size)
                target_network = self.makeRandomNetwork(target_size, target_size)
                result = reference_network.makeCompatibleAssignments(target_network,
                            cn.OR_SPECIES, identity=identity, is_subsets=True, max_num_assignment=100000)
                if len(result.assignment_arr) == 0:
                    continue
                if IGNORE_TEST:
                    print(f"Time: {time.time() - start_time:.4f}", len(result.assignment_arr), result.is_truncated)
                self.assertGreater(len(result.assignment_arr), 0)
                self.assertEqual(len(result.assignment_arr[0]), reference_size)
                break
            else:
                self.assertTrue(False)
            if IGNORE_TEST:
                print(np.sum(np.log10(result.compression_factor)))
        #
        test(cn.ID_WEAK)
        test(cn.ID_STRONG)

    def testIsStructurallyIdenticalBasic(self):
        if IGNORE_TEST:
            return
        def permuteArray(arr, row_perm, column_perm):
            new_arr = arr.copy()
            new_arr = new_arr[row_perm, :]
            new_arr = new_arr[:, column_perm]
            return new_arr
        #
        species_perm = np.array([1, 2, 0])
        reaction_perm = np.array([1, 0])
        reactant_arr = permuteArray(self.network.reactant_mat.values, species_perm, reaction_perm)
        product_arr = permuteArray(self.network.product_mat.values, species_perm, reaction_perm)
        network = Network(reactant_arr, product_arr)
        result = self.network.isStructurallyIdentical(network, identity=cn.ID_WEAK)
        self.assertTrue(result)
        result = self.network.isStructurallyIdentical(self.network, identity=cn.ID_STRONG)
        self.assertTrue(result)

    def testIsStructurallyIdenticalDoubleFail(self):
        if IGNORE_TEST:
            return
        # Double the size of the target. Look for exact match. Should fail.
        reactant_arr = np.vstack([self.network.reactant_mat.values]*2)
        reactant_arr = np.hstack([reactant_arr]*2)
        product_arr = np.vstack([self.network.product_mat.values]*2)
        product_arr = np.hstack([product_arr]*2)
        target_network = Network(reactant_arr, product_arr)
        result = self.network.isStructurallyIdentical(target_network, is_subsets=False, identity=cn.ID_WEAK)
        self.assertFalse(result)
    
    def testIsStructurallyIdenticalDoubleSubset(self):
        if IGNORE_TEST:
            return
        reactant_arr = np.hstack([self.network.reactant_mat.values]*2)
        product_arr = np.hstack([self.network.product_mat.values]*2)
        target_network = Network(reactant_arr, product_arr)
        result = self.network.isStructurallyIdentical(target_network, is_subsets=True, identity=cn.ID_WEAK)
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
        def test(reference_size, target_factor=1, num_iteration=100):
            success_cnt = 0
            total_cnt = 0
            for _ in range(num_iteration):
                for identity in [cn.ID_WEAK, cn.ID_STRONG]:
                    for is_subsets in [True, False]:
                        if (not is_subsets) and (target_factor > 1):
                            continue
                        reference = Network.makeRandomNetworkByReactionType(reference_size)
                        target, assignment_pair = reference.permute()
                        target_reactant_arr = np.hstack([target.reactant_mat.values]*target_factor)
                        target_product_arr = np.hstack([target.product_mat.values]*target_factor)
                        target = Network(target_reactant_arr, target_product_arr)
                        result = reference.isStructurallyIdentical(target, identity=identity, is_subsets=is_subsets,
                              expected_assignment_pair=assignment_pair, max_num_assignment=1000000)
                        total_cnt += 1
                        if result.is_truncated:
                            continue
                        success_cnt += 1
                        self.assertTrue(bool(result))
                        first_matched_network = target.makeNetworkFromAssignmentPair(result.assignment_pairs[0])
                        if (not is_subsets) and (identity == cn.ID_STRONG):
                            first_matched_network = target.makeNetworkFromAssignmentPair(result.assignment_pairs[0])
                            self.assertTrue(np.all(
                                first_matched_network.reactant_mat.values == reference.reactant_mat.values))
                            self.assertTrue(np.all(
                                    first_matched_network.product_mat.values == reference.product_mat.values))
            if IGNORE_TEST:
                print(f"Total count: {total_cnt}; Success count: {success_cnt}; reference_size: {reference_size}; target_factor: {target_factor}")
        #
        for target_factor in [1, 2]:
            for size in [3, 5, 8]:
                test(size, target_factor=target_factor)

    def testIsStructurallyIdenticalScaleRandomlyPermuteFalse(self):
        if IGNORE_TEST:
            return
        def test(reference_size, target_factor=1, num_iteration=1000):
            count = 0
            for _ in range(num_iteration):
                for identity in [cn.ID_STRONG]:
                    for is_subsets in [False]:
                        reference = Network.makeRandomNetworkByReactionType(reference_size)
                        target, _ = reference.permute()
                        target_reactant_arr = np.hstack([target.reactant_mat.values]*target_factor)
                        target_product_arr = np.hstack([target.product_mat.values]*target_factor)
                        target = Network(target_reactant_arr, target_product_arr)
                        # Change the target so that it's no longer structurally identical
                        irow, icolumn = (np.random.randint(0, target.num_species),
                              np.random.randint(0, target.num_reaction))
                        if target.reactant_mat.values[irow, icolumn] == 0:
                            target.reactant_mat.values[irow, icolumn] = 1
                        else:
                            target.reactant_mat.values[irow, icolumn] = 0
                        # Analyze
                        result = reference.isStructurallyIdentical(target, identity=identity, is_subsets=is_subsets)
                        count += 1
                        self.assertFalse(result)
            #print(reference_size, count)
        #
        for size in [5, 10, 20, 40]:
            test(size)

    def testMakeCompatibilityVectorPermutedNetwork(self):
        if IGNORE_TEST:
            return
        identity = cn.ID_WEAK
        is_subsets = False
        for _ in range(10):
            reference = Network.makeRandomNetworkByReactionType(10)
            target, _ = reference.permute()
            # FIXME: Check both reaction and species permutation
            compatibility_vector = reference.makeCompatibilitySetVector(target, identity=identity,
                                                                        is_subsets=is_subsets)
            # Check that sets are not the same size
            std = np.std([len(s) for s in compatibility_vector])
            if std > 0.1:
                break
        else:
            raise RuntimeError("All sets are the same size")
        permuted_network, inversion_permutation = reference._makeCompatibilityVectorPermutedNetwork(target,
              identity=identity, is_subsets=is_subsets)
        original_network, _ = permuted_network.permute(assignment_pair=inversion_permutation)
        self.assertEqual(reference, original_network)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            network = self.makeRandomNetwork(10, 10)
            ser = network.serialize()
            self.assertTrue(isinstance(ser, pd.Series))
            new_network = Network.deserialize(ser)
            self.assertEqual(network, new_network)


if __name__ == '__main__':
    unittest.main(failfast=True)