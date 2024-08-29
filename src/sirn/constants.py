import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEST_DIR = os.path.join(PROJECT_DIR, 'src', 'sirn_tests')
OSCILLATOR_DIRS = [
        "Oscillators_May_28_2024_8898",
        "Oscillators_June_9_2024_14948",
        "Oscillators_June_10_A_11515",
        "Oscillators_June_10_B_10507",
        "Oscillators_June_11_10160",
        "Oscillators_June_11_A_2024_6877",
        "Oscillators_June_11_B_2024_7809",
        "Oscillators_DOE_JUNE_10_17565",
        "Oscillators_DOE_JUNE_12_A_30917",
        "Oscillators_DOE_JUNE_12_B_41373",
        "Oscillators_DOE_JUNE_12_C_27662",
        ]
MAX_NUM_ASSIGNMENT = 10000  # Maximum number of permutations to search
#STRUCTURAL_IDENTITY_TYPE = "structural_identity_type"
#STRUCTURAL_IDENTITY_TYPE_NOT = "identity_collection_type_not"
#STRUCTURAL_IDENTITY_TYPE_WEAK = "identity_collection_type_weak"
#STRUCTURAL_IDENTITY_TYPE_STRONG = "identity_collection_type_strong"
UNKNOWN_STRUCTURAL_IDENTITY_NAME = "*" # Used for networks whose structural identity cannot be determined
NETWORK_NAME_DELIMITER = "---"
NETWORK_NAME_PREFIX_KNOWN = "!"
NETWORK_NAME_PREFIX_UNKNOWN = "?"
NETWORK_NAME_SUFFIX = "_"
STRUCTURAL_IDENTITY_PREFIX_STRONG = "+"
STRUCTURAL_IDENTITY_PREFIX_WEAK = "-"
NUM_HASH = 'num_hash'
MAX_HASH = 'max_hash'
NON_SIRN_HASH = 1
# Numerical constants
# Network matrices
#   Matrix type
MT_STOICHIOMETRY = 'mt_stoichiometry'
MT_SINGLE_CRITERIA = 'mt_single_criteria'
MT_PAIR_CRITERIA = 'mt_pair_criteria'
MT_LST = [MT_STOICHIOMETRY, MT_SINGLE_CRITERIA, MT_PAIR_CRITERIA]
#   Orientation
OR_REACTION = 'or_reaction'
OR_SPECIES = 'or_species'
OR_LST = [OR_REACTION, OR_SPECIES]
#   Participant
PR_REACTANT = 'pr_reactant'
PR_PRODUCT = 'pr_product'
PR_LST = [PR_REACTANT, PR_PRODUCT]
#   Identity
ID_WEAK = 'id_weak'
ID_STRONG = 'id_strong'
ID_LST = [ID_WEAK, ID_STRONG]
# Network serialization format
NETWORK_NAME = 'network_name'
REACTANT_ARRAY_STR = 'reactant_array_str'
PRODUCT_ARRAY_STR = 'product_array_str'
NUM_SPECIES = 'num_species'
NUM_REACTION = 'num_reaction'
SPECIES_NAMES = 'species_names'
REACTION_NAMES = 'reaction_names'
CRITERIA_ARRAY_STR = 'boundary_array_str'
CRITERIA_ARRAY_LEN = 'boundary_array_len'
SERIALIZATION_NAMES = [NETWORK_NAME, REACTANT_ARRAY_STR, PRODUCT_ARRAY_STR, SPECIES_NAMES,
                       REACTION_NAMES, NUM_SPECIES, NUM_REACTION]
CRITERIA_BOUNDARY_VALUES = [-1.0, 0.0, 1.0]