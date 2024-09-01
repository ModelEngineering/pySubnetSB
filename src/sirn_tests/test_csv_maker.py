import sirn.constants as cn # type: ignore
from sirn.csv_maker import CSVMaker # type: ignore
from sirn.assignment_pair import AssignmentPair, AssignmentCollection  # type: ignore

import time
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
TYPE1_DCT = {"first": int, "second": float, "third": np.ndarray}
TYPE2_DCT = {"fourth": np.ndarray, "fifth": AssignmentPair, "sixth": AssignmentCollection}
TYPE_DCT = dict(TYPE1_DCT)
TYPE_DCT.update(TYPE2_DCT)
ASSIGNMENT_PAIR = AssignmentPair(species_assignment=np.array(range(3)),reaction_assignment=np.array(range(5)))
VALUES_DCT = {"first": 1, "second": 0.1, "third": np.array([0, 1]), "fourth": np.array([ [0, 1], [1, 0]]),
              "fifth": ASSIGNMENT_PAIR, "sixth": AssignmentCollection([ASSIGNMENT_PAIR]*3)}


#############################
# Tests
#############################
class TestCSVMaker(unittest.TestCase):

    def setUp(self):
        self.csv_maker = CSVMaker(TYPE_DCT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.csv_maker.names == [p for p in TYPE_DCT.keys()]))

    def testRepr(self):
        if IGNORE_TEST:
            return
        stg = str(self.csv_maker)
        self.assertTrue(all([s in stg for s in TYPE_DCT.keys()]))

    def testEncodeDecode(self):
        if IGNORE_TEST:
            return
        encoding = self.csv_maker.encode(**VALUES_DCT)
        dct = self.csv_maker.decode(encoding)
        trues = [np.all(VALUES_DCT[n] == dct[n]) for n in VALUES_DCT.keys()]
        self.assertTrue(all(trues))

    def testAppend(self):
        if IGNORE_TEST:
            return
        csv_maker1 = CSVMaker(TYPE1_DCT)
        csv_maker2 = CSVMaker(TYPE2_DCT)
        csv_maker = csv_maker1.append(csv_maker2)
        self.assertTrue(csv_maker == self.csv_maker)

    def testEncodeDecodeAssignmentCollection(self):
        if IGNORE_TEST:
            return
        size = 3
        assignment_pair = AssignmentPair(species_assignment=np.array(range(3)),
              reaction_assignment=np.array(range(5)))
        assignment_pairs = AssignmentCollection([assignment_pair]*size)
        encoding = self.csv_maker._encodeAssignmentCollection(assignment_pairs)
        assignment_pairs2 = self.csv_maker._decodeAssignmentCollection(encoding)
        for ap1, ap2 in zip(assignment_pairs.pairs, assignment_pairs2.pairs):
            self.assertEqual(ap1, ap2)

if __name__ == '__main__':
    unittest.main()