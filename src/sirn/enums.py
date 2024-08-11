'''Enums used in package.'''

# Base class for enums
class Enum(object):
    PERMITTED_STRS:list = []  # Must override this

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)
    
    def __eq__(self, other):
        return str(self) == str(other)
    
    def hasValue(self, value:str):
        return str(self.__class__) == value
    
# Derived classes
class NoneEnum(Enum):
    def __init__(self, value):
        self.value = "None"

class ParticipantEnum(Enum):
    REACTANT = 'reactant'
    PRODUCT = 'product'
    PERMITTED_STRS = [REACTANT, PRODUCT]

class OrientationEnum(Enum):
    REACTION = 'reaction'
    SPECIES = 'species'
    PERMITTED_STRS = [REACTION, SPECIES]

class IdentityEnum(Enum):
    WEAK = 'weak'
    STRONG = 'strong'
    PERMITTED_STRS = [WEAK, STRONG]

# Matrix type
class MatrixTypeEnum(Enum):
    STANDARD = 'standard'
    SINGLE_CRITERIA = 'single_criteria'
    PAIR_CRITERIA = 'pair_criteria'
    PERMITTED_STRS = [STANDARD, SINGLE_CRITERIA, PAIR_CRITERIA]