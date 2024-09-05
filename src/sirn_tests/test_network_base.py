import sirn.constants as cn  # type: ignore
from sirn.network_base import NetworkBase # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import numpy as np
import copy
import tellurium as te  # type: ignore
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
NETWORK = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = copy.deepcopy(NETWORK)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        for mat in [self.network.reactant_mat, self.network.product_mat]:
            self.assertTrue(isinstance(mat, NamedMatrix))
        self.assertTrue("int" in str(type(self.network.weak_hash)))
        self.assertTrue("int" in str(type(self.network.strong_hash)))

    def testGetNetworkMatrix(self):
        if IGNORE_TEST:
            return
        reactant_mat = self.network.getNetworkMatrix(
               matrix_type=cn.MT_STOICHIOMETRY,
               orientation=cn.OR_SPECIES,
               participant=cn.PR_REACTANT,
                identity=cn.ID_STRONG)
        self.assertTrue(isinstance(reactant_mat, NamedMatrix))
        df = reactant_mat.dataframe
        self.assertTrue(df.columns.name == "reactions")
        self.assertTrue(df.index.name == "species")
    
    def testGetNetworkMatrixMultiple(self):
        if IGNORE_TEST:
            return
        count = 0
        for i_matrix_type in cn.MT_LST:
                for i_orientation in cn.OR_LST:
                    for i_identity in cn.ID_LST:
                        for i_participant in cn.PR_LST:
                            if i_identity == cn.ID_WEAK:
                                if i_participant == cn.PR_PRODUCT:
                                    continue
                                else:
                                    i_participant = None
                            mat = self.network.getNetworkMatrix(i_matrix_type, i_orientation, i_participant, i_identity)
                            count += 1
                            if i_matrix_type == cn.MT_STOICHIOMETRY:
                                self.assertTrue(isinstance(mat, NamedMatrix))
                                if i_orientation == cn.OR_SPECIES:
                                    self.assertTrue(mat.dataframe.index.name == "species")
                                else:
                                    self.assertTrue(mat.dataframe.index.name == "reactions")
                            elif i_matrix_type == cn.MT_SINGLE_CRITERIA:
                                self.assertTrue(isinstance(mat, SingleCriteriaCountMatrix))
                            elif i_matrix_type == cn.MT_PAIR_CRITERIA:
                                self.assertTrue(isinstance(mat, PairCriteriaCountMatrix))
                            else:
                                self.assertTrue(False)
        self.assertEqual(count, 18)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        network = self.network.copy()
        self.assertEqual(self.network, network)
        
    def testRandomlyPermuteTrue(self):
        if IGNORE_TEST:
            return
        def test(size, num_iteration=500):
            reactant_arr = np.random.randint(0, 3, (size, size))
            product_arr = np.random.randint(0, 3, (size, size))
            network = NetworkBase(reactant_arr, product_arr)
            for _ in range(num_iteration):
                new_network, assignment_pair = network.permute()
                if network == new_network:
                    continue
                original_network, _ = new_network.permute(assignment_pair=assignment_pair)
                self.assertTrue(network.isEquivalent(original_network))
                self.assertEqual(network.num_species, new_network.num_species)
                self.assertEqual(network.num_reaction, new_network.num_reaction)
                self.assertEqual(network._weak_hash, new_network._weak_hash)
                self.assertEqual(network.strong_hash, new_network.strong_hash)
        #
        test(3)
        test(30)

    def testRandomlyPermuteFalse(self):
        if IGNORE_TEST:
            return
        # Randomly change a value in the reactant matrix
        def test(size, num_iteration=500):
            reactant_arr = np.random.randint(0, 3, (size, size))
            product_arr = np.random.randint(0, 3, (size, size))
            network = NetworkBase(reactant_arr, product_arr)
            weak_collision_cnt = 0
            strong_collision_cnt = 0
            for _ in range(num_iteration):
                new_network, _ = network.permute()
                irow = np.random.randint(0, size)
                icol = np.random.randint(0, size)
                cur_value = new_network.reactant_mat.values[irow, icol]
                if cur_value == 0:
                    new_network.reactant_mat.values[irow, icol] = 1
                else:
                    new_network.reactant_mat.values[irow, icol] = 0
                self.assertNotEqual(network, new_network)
                self.assertEqual(network.num_species, new_network.num_species)
                self.assertEqual(network.num_reaction, new_network.num_reaction)
                if network.weak_hash == new_network.weak_hash:
                    weak_collision_cnt += 1
                if network.strong_hash == new_network.strong_hash:
                    strong_collision_cnt += 1
            frac_weak = weak_collision_cnt/num_iteration
            frac_strong = strong_collision_cnt/num_iteration
            self.assertTrue(frac_weak < 0.1)
            self.assertTrue(frac_strong < 0.1)
            return frac_weak, frac_strong
        #
        test(30)

    def testIsStructurallyCompatible(self):
        if IGNORE_TEST:
            return
        network1 = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)
        network2 = NetworkBase.makeFromAntimonyStr(NETWORK2, network_name=NETWORK_NAME)
        network3 = NetworkBase.makeFromAntimonyStr(NETWORK3, network_name=NETWORK_NAME)
        network4 = NetworkBase.makeFromAntimonyStr(NETWORK4, network_name=NETWORK_NAME)
        self.assertFalse(network1.isStructurallyCompatible(network4))
        self.assertTrue(network1.isStructurallyCompatible(network2))
        self.assertTrue(network1.isStructurallyCompatible(network3, identity=cn.ID_WEAK))

    def testPrettyPrintReaction(self):
        if IGNORE_TEST:
            return
        network = NetworkBase.makeFromAntimonyStr(NETWORK3, network_name="Network3")
        stg = network.prettyPrintReaction(0)
        self.assertTrue("2.0 S1 -> 2.0 S1 + S2" in stg)

    def testIsMatrixEqual(self):
        if IGNORE_TEST:
            return
        network = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)
        network2 = NetworkBase.makeFromAntimonyStr(NETWORK2, network_name=NETWORK_NAME)
        network3 = NetworkBase.makeFromAntimonyStr(NETWORK3, network_name=NETWORK_NAME)
        self.assertTrue(network.isMatrixEqual(network))
        self.assertTrue(network.isMatrixEqual(network2))
        self.assertFalse(network.isMatrixEqual(network3, identity=cn.ID_STRONG))

    def testHash(self):
        if IGNORE_TEST:
            return
        size = 5
        def makeHashArray(network):
            single_criteria_matrices = [network.getNetworkMatrix(
                                  matrix_type=cn.MT_SINGLE_CRITERIA, orientation=o,
                                  identity=cn.ID_WEAK)
                                  for o in [cn.OR_SPECIES, cn.OR_REACTION]]
            this_hash_arr = np.array([s.row_order_independent_hash for s in single_criteria_matrices])
            return this_hash_arr
        network = NetworkBase.makeRandomNetwork(size, size)
        other, _ = network.permute()
        this_hash_arr = makeHashArray(network)
        other_hash_arr = makeHashArray(other)
        self.assertTrue(np.all(this_hash_arr == other_hash_arr))

    def testMakeRandomNetworkFromReactionType(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            size = np.random.randint(3, 20)
            network = NetworkBase.makeRandomNetworkByReactionType(size)
            eval_arr = np.hstack([network.reactant_mat.values, network.product_mat.values])
            sum_arr = np.sum(eval_arr, axis=1)
            self.assertTrue(np.all(sum_arr > 0))

    def testToFromSeries(self):
        #if IGNORE_TEST:
        #    return
        series = self.network.toSeries()
        serialization_str = self.network.seriesToJson(series)
        network = NetworkBase.deserialize(serialization_str)
        self.assertTrue(self.network == network)

if __name__ == '__main__':
    unittest.main(failfast=True)