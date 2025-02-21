import pySubnetSB.api as api # type: ignore
from pySubnetSB.network import Network  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import os
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = True
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
SERIALIZATION_PATH = os.path.join(cn.TEST_DIR, "test_api.txt")
REMOVE_FILES = [SBML_PATH, ANT_PATH, SERIALIZATION_PATH]
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")

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

    @staticmethod
    def optionIter(excludes=None):
        if excludes is None:
            excludes = []
        dct = {}
        dcts = []
        for is_subnet in [True, False]:
            if not 'is_subnet' in excludes:
                dct['is_subnet'] = is_subnet
            for num_process in [1, -1]:
                if not 'num_process' in excludes:
                    dct['num_process'] = num_process
                for max_num_assignment in [1e10, 1]:
                    if not 'max_num_assignment' in excludes:
                        dct['max_num_assignment'] = max_num_assignment
                    for identity in [cn.ID_STRONG, cn.ID_WEAK]:
                        if not 'identity' in excludes:
                            dct['identity'] = identity
                        is_duplicate = False
                        for dct1 in dcts:
                            if dct1 == dct:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            yield dct

    def testOptionIter(self):
        if IGNORE_TEST:
            return
        for dct in self.optionIter(["is_subnet", "num_process"]):
            self.assertFalse("num_process" in dct)

    def testFindReferenceInTarget(self):
        if IGNORE_TEST:
            return
        iter = self.optionIter()
        for dct in iter:
            result = api.findReferenceInTarget(MODEL, MODEL, **dct)
            self.assertTrue(len(result.assignment_pairs) == 1)
            self.assertTrue(np.all(result.assignment_pairs[0].species_assignment == [0, 1]))
            self.assertTrue(np.all(result.assignment_pairs[0].reaction_assignment == [0]))
            self.assertTrue(result.is_truncated == False)

    def testClusterStructurallyIdenticalModelsInDirectory(self):
        if IGNORE_TEST:
            return
        ffiles = [f for f in os.listdir(MODEL_DIR) if "best" in f]
        iter = self.optionIter(["is_subnet", "num_process"])
        for dct in iter:
            df = api.clusterStructurallyIdenticalModelsInDirectory(MODEL_DIR, **dct)
            self.assertEqual(len(df), len(ffiles))

    def testFindReferencesInTargets(self):
        if IGNORE_TEST:
            return
        count = len([f for f in os.listdir(MODEL_DIR) if "best" in f])
        iter = self.optionIter(["is_subnet"])
        for dct in iter:
            df = api.findReferencesInTargets(MODEL_DIR, MODEL_DIR, **dct)
            self.assertEqual(len(df), count**2)
            num_match = np.sum([len(v) > 0 for v in df[cn.FINDER_INDUCED_NETWORK]]) \
                + np.sum([v for v in df[cn.FINDER_IS_TRUNCATED]])
            self.assertGreaterEqual(num_match, count)  # May be networks that are a subnet of another

    def testMakeSerializationFile(self):
        if IGNORE_TEST:
            return
        api.makeSerializationFile(MODEL_DIR, SERIALIZATION_PATH, is_report=IS_PLOT)
        self.assertTrue(os.path.isfile(SERIALIZATION_PATH))

    def testGetNetworkCollection(self):
        #if IGNORE_TEST:
        #    return
        network_collection = api._getNetworkCollection(MODEL_DIR)
        self.assertTrue(len(network_collection) > 0)
        #
        url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt"
        network_collection = api._getNetworkCollection(url)
        self.assertTrue(len(network_collection) > 0)


if __name__ == '__main__':
    unittest.main()