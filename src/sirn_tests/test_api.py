import sirn.api as api # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn  # type: ignore

import os
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MODEL = """
J1: A -> B; k1*A

k1 = 0.1
A = 10
B = 0
"""
MODEL_RR = te.loada(MODEL)
SBML_PATH = os.path.join(cn.TEST_DIR, "test_api.sbml")
ANT_PATH = os.path.join(cn.TEST_DIR, "test_api.ant")
REMOVE_FILES = [SBML_PATH, ANT_PATH]

#############################
# Tests
#############################

class TestModelSpecification(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.specification = api.ModelSpecification(MODEL, specification_type="antstr")

    def tearDown(self):
        self.remove()

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.isfile(path):
                os.remove(path)

    def testMakeNetworkAntimonyStr(self):
        if IGNORE_TEST:
            return
        network = self.specification.makeNetwork(MODEL)
        self.assertTrue(np.all(network.species_names == ["A", "B"]))
        self.assertTrue(network.reaction_names == ["J1"])
    
    def testMakeNetworkAntimonyFile(self):
        if IGNORE_TEST:
            return
        with open(ANT_PATH, "w") as fd:
            fd.write(MODEL)
        network = self.specification.makeNetwork(ANT_PATH, specification_type="antfile")
        self.assertTrue(np.all(network.species_names == ["A", "B"]))
        self.assertTrue(network.reaction_names == ["J1"])

    def testMakeNetworkSBMLStr(self):
        if IGNORE_TEST:
            return
        sbml_str = MODEL_RR.getSBML()
        network = api.ModelSpecification.makeNetwork(sbml_str, specification_type="sbmlstr")
        self.assertTrue(np.all(network.species_names == ["A", "B"]))
        self.assertTrue(network.reaction_names == ["J1"])

    def testMakeNetworkSBMLFile(self):
        if IGNORE_TEST:
            return
        sbml_str = MODEL_RR.getSBML()
        with open(SBML_PATH, "w") as fd:
            fd.write(sbml_str)
        network = self.specification.makeNetwork(SBML_PATH, specification_type="sbmlfile")
        self.assertTrue(np.all(network.species_names == ["A", "B"]))
        self.assertTrue(network.reaction_names == ["J1"])
        

class TestFunctions(unittest.TestCase):

    def testFindReferenceInTarget(self):
        if IGNORE_TEST:
            return
        result = api.findReferenceInTarget(MODEL, MODEL, is_report=IS_PLOT)
        self.assertTrue(len(result.assignment_pairs) == 1)
        self.assertTrue(np.all(result.assignment_pairs[0].species_assignment == [0, 1]))
        self.assertTrue(np.all(result.assignment_pairs[0].reaction_assignment == [0]))
        self.assertTrue(result.is_truncated == False)

    def testClusterStructurallyIdenticalModelsInDirectory(self):
        if IGNORE_TEST:
            return
        DIR = os.path.join(cn.TEST_DIR, "oscillators")
        ffiles = [f for f in os.listdir(DIR) if "best" in f]
        df = api.clusterStructurallyIdenticalModelsInDirectory(DIR, cn.ID_STRONG, is_report=IS_PLOT)
        self.assertEqual(len(df), len(ffiles))

    def testFindReferencesInTargets(self):
        if IGNORE_TEST:
            return
        DIR = os.path.join(cn.TEST_DIR, "oscillators")
        count = len([f for f in os.listdir(DIR) if "best" in f])
        df = api.findReferencesInTargets(DIR, DIR, cn.ID_STRONG, is_report=IS_PLOT)
        self.assertEqual(len(df), count**2)
        num_match = np.sum([len(v) > 0 for v in df["induced_network"]])
        self.assertGreaterEqual(num_match, count)  # May be networks that are a subnet of another


if __name__ == '__main__':
    unittest.main()