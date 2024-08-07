from sirn import enums  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False


#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def testEnumOrientation(self):
        if IGNORE_TEST:
            return
        orientation = enums.OrientationEnum('reaction')
        self.assertEqual(str(orientation), 'reaction')
        #
        new_orientation = enums.OrientationEnum('reaction')
        self.assertTrue(orientation == new_orientation)
        #
        new_orientation = enums.OrientationEnum('species')
        self.assertTrue(orientation != new_orientation)

    def testOthers(self):
        for cls in [enums.ParticipantEnum, enums.IdentityEnum, enums.MatrixTypeEnum]:
            enum1 = cls(cls.PERMITTED_STRS[0])
            enum2 = cls(cls.PERMITTED_STRS[0])
            self.assertTrue(enum1 == enum2)
            #
            enum2 = cls(cls.PERMITTED_STRS[1])
            self.assertTrue(enum1 != enum2)


if __name__ == '__main__':
    unittest.main()