'''Enums used in package.'''

# Base class for enums
class Enum(object):
    PERMITTED_STRS:list = []  # Must override this

    def __init__(self, value):
        self.value = self.PERMITTED_STRS.index(value)

    def __repr__(self):
        return self.PERMITTED_STRS[self.value]
    
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            raise RuntimeError("Cannot compare different enums.")
        return self.value == other.value
    
# Derived classes
class ParticipantEnum(Enum):
    PERMITTED_STRS = ['reactant', 'product']

class OrientationEnum(Enum):
    PERMITTED_STRS = ['reaction', 'species']

class IdentityEnum(Enum):
    PERMITTED_STRS = ['weak', 'strong']

# Matrix type
class MatrixTypeEnum(Enum):
    PERMITTED_STRS = ['stoichiometry', 'criteria', 'criteria_pair']