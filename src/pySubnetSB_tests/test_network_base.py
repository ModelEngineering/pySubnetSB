import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.network_base import NetworkBase # type: ignore
from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB import util # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair # type: ignore

import os
from pynauty import Graph  # type: ignore
import matplotlib.pyplot as plt
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

    @util.timeit
    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        for mat in [self.network.reactant_nmat, self.network.product_nmat]:
            self.assertTrue(isinstance(mat, NamedMatrix))
        self.assertTrue("int" in str(type(self.network.network_hash)))
        self.assertTrue("int" in str(type(self.network.network_hash)))
    
    @util.timeit
    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        network = self.network.copy()
        self.assertEqual(self.network, network)

    @util.timeit
    def testIsMatrixEqual(self):
        if IGNORE_TEST:
            return
        network = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)
        network3 = NetworkBase.makeFromAntimonyStr(NETWORK3, network_name=NETWORK_NAME)
        self.assertTrue(network.isMatrixEqual(network3, identity=cn.ID_WEAK))
        self.assertFalse(network.isMatrixEqual(network3, identity=cn.ID_STRONG))
        self.assertTrue(network.isMatrixEqual(network, identity=cn.ID_WEAK))
        self.assertTrue(network.isMatrixEqual(network, identity=cn.ID_STRONG))
        
    @util.timeit
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
                self.assertEqual(network.network_hash, new_network.network_hash)
                self.assertEqual(network.network_hash, new_network.network_hash)
        #
        test(3)
        test(30)

    @util.timeit
    def testRandomlyPermuteFalse(self):
        if IGNORE_TEST:
            return
        # Randomly change a value in the reactant matrix
        def test(size, num_iteration=500):
            network_hashes = []
            for _ in range(num_iteration):
                network = NetworkBase.makeRandomNetworkByReactionType(num_reaction=3*size, num_species=size)
                network_hashes.append(network.network_hash)
            num_distinct_weak_hashes = len(set(network_hashes))
            frac_distinct_hashes =  num_distinct_weak_hashes/ num_iteration
            #print(f"distinct hashes: {frac_distinct_hashes}")
            self.assertGreater(frac_distinct_hashes, 0.9)
        #
        test(4)

    @util.timeit
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

    @util.timeit
    def testPrettyPrintReaction(self):
        if IGNORE_TEST:
            return
        network = NetworkBase.makeFromAntimonyStr(NETWORK3, network_name="Network3")
        stg = network.prettyPrintReaction(0)
        self.assertTrue("2.0 S1 -> 2.0 S1 + S2" in stg)

    @util.timeit
    def testMakeRandomNetworkFromReactionType(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            size = np.random.randint(3, 20)
            network = NetworkBase.makeRandomNetworkByReactionType(size, is_exact=True)
            self.assertEqual(network.num_species, size)
            eval_arr = np.vstack([network.reactant_nmat.values, network.product_nmat.values])
            # Verify that all reactions have at least one participant
            sum_arr = np.sum(eval_arr, axis=0)
            self.assertTrue(np.all(sum_arr > 0))

    @util.timeit
    def testToFromSeries(self):
        if IGNORE_TEST:
            return
        series = self.network.toSeries()
        serialization_str = self.network.seriesToJson(series)
        network = NetworkBase.deserialize(serialization_str)
        self.assertTrue(self.network == network)

    @util.timeit
    def testMakePynautyNetwork(self):
        if IGNORE_TEST:
            return
        graph = self.network.makePynautyNetwork()
        self.assertTrue(isinstance(graph, Graph))

    @util.timeit
    def testCSVNetwork(self):
        if IGNORE_TEST:
            return
        csv_format = self.network.makeCSVNetwork()
        self.assertTrue(isinstance(csv_format, str))
        num_line = csv_format.count("\n")
        num_sep = csv_format.count(">")
        self.assertEqual(num_line+1,num_sep + self.network.num_species + self.network.num_reaction)
    
    @util.timeit
    def testGetGraphDct(self):
        if IGNORE_TEST:
            return
        num_iteration = 100
        size =10 
        for _ in range(num_iteration):
            for identity in cn.ID_LST:
                network = NetworkBase.makeRandomNetwork(size, size)
                graph_descriptor = network.getGraphDescriptor(identity=identity)
                vertex_dct = graph_descriptor.vertex_dct
                label_dct = graph_descriptor.label_dct
                self.assertTrue(isinstance(vertex_dct, dict))
                self.assertTrue(isinstance(vertex_dct[0], list))
                if identity == cn.ID_STRONG:
                    num_edge = network.reactant_nmat.values.sum() + network.product_nmat.values.sum()
                else:
                    num_edge = np.abs(network.standard_nmat.values).sum()
                num_graph_edge = np.sum([len(e) for e in vertex_dct.values()])
                self.assertEqual(num_edge, num_graph_edge)
                #
                num_species = np.sum([v == 'species' for v in label_dct.values()])
                self.assertEqual(num_species, network.num_species)
                num_reaction = np.sum([v == 'reaction' for v in label_dct.values()])
                self.assertEqual(num_reaction, network.num_reaction)

    @util.timeit
    def testFill(self):
        if IGNORE_TEST:
            return
        for fill_size in range(1, 5):
            network = NetworkBase.makeRandomNetworkByReactionType(5, 5)
            filled_network = network.fill(num_fill_reaction=fill_size, num_fill_species=fill_size)
            self.assertLessEqual(filled_network.num_species, network.num_species + fill_size)
            self.assertEqual(filled_network.num_reaction, network.num_reaction + fill_size)
        #
        with self.assertRaises(ValueError):
            network.fill(num_fill_reaction=0, num_fill_species=0)

    @util.timeit

    def testIsBoundaryNetwork(self):
        if IGNORE_TEST:
            return
        boundary_network = """
            R1: A -> ; k1*A
            R2:  -> B; k2
            k1 = 0.1; k2 = 0.2
            A = 0; B = 0
        """
        network = NetworkBase.makeFromAntimonyStr(boundary_network)
        self.assertTrue(network.isBoundaryNetwork())
        #
        boundary_network = """
            R1: A -> ; k1*A
            R3: A -> C; k1*A
            R2: B -> A; k2*B
            k1 = 0.1; k2 = 0.2
            A = 0; B = 0
        """
        network = NetworkBase.makeFromAntimonyStr(boundary_network)
        self.assertFalse(network.isBoundaryNetwork())

    def testMakeFromAntimonyStrRoadrunner(self):
        if IGNORE_TEST:
            return
        roadrunner = te.loada(BIG_NETWORK)
        network = NetworkBase.makeFromAntimonyStr(None, roadrunner=roadrunner)
        self.assertGreater(network.num_species, 0)

    def testMakeFromSBMLFile(self):
        if IGNORE_TEST:
            return
        PATH = os.path.join(cn.TEST_DIR, "xml_files/BIOMD0000000033.xml")
        network = NetworkBase.makeFromSBMLFile(PATH)
        self.assertGreater(network.num_species, 0)

    def testMakeInducedNetwork(self):
        if IGNORE_TEST:
            return
        species_assignment = np.array(range(self.network.num_species))
        reaction_assignment = np.array(range(self.network.num_reaction))
        assignment_pair = AssignmentPair(species_assignment, reaction_assignment)
        induced_network = self.network.makeInducedNetwork(assignment_pair)
        self.assertTrue(self.network.isEquivalent(induced_network))

    def testMakeMatricesForIdentity(self):
        if IGNORE_TEST:
            return
        network1 = NetworkBase.makeFromAntimonyStr(NETWORK1, network_name=NETWORK_NAME)
        network4 = NetworkBase.makeFromAntimonyStr(NETWORK4, network_name=NETWORK_NAME)
        #####
        def test(network, identity, expected_bool):
            reactant_nmat, product_nmat = network.makeMatricesForIdentity(identity=identity)
            self.assertEqual(reactant_nmat == network.reactant_nmat, expected_bool)
            self.assertEqual(product_nmat == network.product_nmat, expected_bool)
        #####
        test(network1, cn.ID_STRONG, True)
        test(network1, cn.ID_WEAK, False)
        test(network4, cn.ID_STRONG, True)
        test(network4, cn.ID_WEAK, True)


if __name__ == '__main__':
    unittest.main(failfast=False)