import numpy as np
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
# Oscillator project
OSCILLATOR_PROJECT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    OSCILLATOR_PROJECT = os.path.dirname(OSCILLATOR_PROJECT)
OSCILLATOR_PROJECT = os.path.join(OSCILLATOR_PROJECT, 'OscillatorDatabase')
#
MAX_NUM_ASSIGNMENT = 1e8  # Maximum number of permutations to search
STRUCTURAL_IDENTITY = "structural_identity"
UNKNOWN_STRUCTURAL_IDENTITY_NAME = "*" # Used for networks whose structural identity cannot be determined
NETWORK_DELIMITER = "---"
NETWORK_NAME_PREFIX_KNOWN = "!"
NETWORK_NAME_PREFIX_UNKNOWN = "?"
NETWORK_NAME_SUFFIX = "_"
IDENTITY_PREFIX_STRONG = "+"
IDENTITY_PREFIX_WEAK = "-"
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
""" NETWORK_NAME = 'network_name'
REACTANT_ARRAY_STR = 'reactant_array_str'
PRODUCT_ARRAY_STR = 'product_array_str'
NUM_SPECIES = 'num_species'
NUM_REACTION = 'num_reaction'
SPECIES_NAMES = 'species_names'
REACTION_NAMES = 'reaction_names'
CRITERIA_ARRAY_STR = 'boundary_array_str'
CRITERIA_ARRAY_LEN = 'boundary_array_len'
SERIALIZATION_NAMES = [NETWORK_NAME, REACTANT_ARRAY_STR, PRODUCT_ARRAY_STR, SPECIES_NAMES,
                       REACTION_NAMES, NUM_SPECIES, NUM_REACTION] """
CRITERIA_BOUNDARY_VALUES = [-2, -1, 0, 1, 2]
# Serialization
S_ANTIMONY_DIRECTORY = "s_antimony_directory"
S_ASSIGNMENT_COLLECTION = 's_assignment_collection'
S_ASSIGNMENT_PAIR = 's_assignment_pair'
S_BOUNDARY_VALUES = "s_boundary_values"
S_PROCESSED_NETWORKS = "s_processed_networks"
S_COLUMN_DESCRIPTION = "s_column_description"
S_COLUMN_NAMES = "s_column_names"
S_CRITERIA_VECTOR = "s_criteria_vector"
S_DIRECTORY = "s_directory"
S_HASH_VAL = "s_hash_val"
S_ID = "s_id"  # Class being serialized
S_IDENTITY = "s_identity"
S_IS_INDETERMINATE = 's_is_indeterminate'
S_MODEL_NAME = "s_model_name"
S_NETWORKS = "s_networks"
S_NETWORK_NAME = "s_network_name"
S_NUM_REACTION = "s_num_reaction"
S_NUM_SPECIES = "s_num_species"
S_PROCESSING_TIME = 's_processing_time'
S_PRODUCT_LST = "s_product_lst"
S_PRODUCT_NMAT = "s_product_nmat"  # Named matrix
S_REACTANT_LST = "s_reactant_lst"
S_REACTANT_NMAT = "s_reactant_NMAT"
S_REACTION_ASSIGNMENT_LST = "s_reaction_assignment_lst"
S_REACTION_NAMES = "s_reaction_names"
S_ROW_DESCRIPTION = "s_row_description"
S_ROW_NAMES = "s_row_names"
S_REFERENCE = "s_reference"
S_SPECIES_ASSIGNMENT_LST = "s_species_assignment_lst"
S_SPECIES_NAMES = "s_species_names"
S_TARGET = "s_target"
S_VALUES = "s_values"