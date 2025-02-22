import numpy as np
import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
    if ("pySubnetSB" in PROJECT_DIR) and (not "src" in PROJECT_DIR):
        break
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
EXAMPLE_DIR = os.path.join(PROJECT_DIR, 'examples')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
AUXILIARY_DATA_DIR = os.path.join(PROJECT_DIR, 'auxiliary_data')
SCRIPT_DIR = os.path.join(PROJECT_DIR, 'scripts')
TEST_DIR = os.path.join(PROJECT_DIR, 'src', 'pySubnetSB_tests')
PLOT_DIR = os.path.join(PROJECT_DIR, 'plots')
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
# File names
SERIALIZATION_FILE = "collection_serialization.txt"
# Oscillator project
OSCILLATOR_PROJECT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    OSCILLATOR_PROJECT = os.path.dirname(OSCILLATOR_PROJECT)
OSCILLATOR_PROJECT = os.path.join(OSCILLATOR_PROJECT, 'OscillatorDatabase')
# Constants
NONE = "none"
MAX_NUM_ASSIGNMENT = 1e7  # Maximum number of permutations to search
MAX_BATCH_SIZE = int(1e7)  # Matrix memory in bytes used in a comparison batch in AssignmentEvaluator
STRUCTURAL_IDENTITY = "structural_identity"
UNKNOWN_STRUCTURAL_IDENTITY_NAME = "*" # Used for networks whose structural identity cannot be determined
NETWORK_DELIMITER = "---"
NETWORK_NAME_PREFIX_KNOWN = "!"
NETWORK_NAME_PREFIX_UNKNOWN = "?"
NETWORK_NAME_SUFFIX = "_"
NULL_STR = ""
IDENTITY_PREFIX_STRONG = "+"
IDENTITY_PREFIX_WEAK = "-"
NUM_HASH = 'num_hash'
MAX_HASH = 'max_hash'
OTHER_HASH = 1
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
CRITERIA_ARRAY_STR = 'boundary_array_str'
CRITERIA_ARRAY_LEN = 'boundary_array_len'
SERIALIZATION_NAMES = [NETWORK_NAME, REACTANT_ARRAY_STR, PRODUCT_ARRAY_STR, SPECIES_NAMES,
                       REACTION_NAMES, NUM_SPECIES, NUM_REACTION] """
SPECIES_NAMES = 'species_names'
REACTION_NAMES = 'reaction_names'
CRITERIA_BOUNDARY_VALUES = [-2, -1, 0, 1, 2]
# Serialization
S_ANTIMONY_DIRECTORY = "s_antimony_directory"
S_ASSIGNMENT_COLLECTION = 's_assignment_collection'
S_ASSIGNMENT_PAIR = 's_assignment_pair'
S_BOUNDARY_VALUES = "s_boundary_values"
S_PROCESSED_NETWORKS = "s_processed_networks"
S_COLUMN_DESCRIPTION = "s_column_description"
S_COLUMN_NAMES = "s_column_names"
S_CONSTRAINT_PAIR_DCT = 's_constraint_pair_dct'
S_CRITERIA_VECTOR = "s_criteria_vector"
S_DICT = "s_dict"
S_DIRECTORY = "s_directory"
S_HASH_VAL = "s_hash_val"
S_ID = "s_id"  # Class being serialized
S_IDENTITY = "s_identity"
S_IS_INDETERMINATE = 's_is_indeterminate'
S_MODEL_NAME = "s_model_name"
S_NETWORKS = "s_networks"
S_NETWORK_NAME = "s_network_name"
S_NUM_COLUMN = "s_num_column"
S_NUM_REACTION = "s_num_reaction"
S_NUM_ROW = "s_num_row"
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
# Finder dataframe columns
FINDER_WORKER_IDX = "worker_idx"
FINDER_REFERENCE_NAME = "reference_name"
FINDER_REFERENCE_IDX = "reference_idx"
FINDER_TARGET_NAME = "target_name"
FINDER_TARGET_IDX = "target_idx"
FINDER_REFERENCE_NETWORK = "reference_network"
FINDER_INDUCED_NETWORK = "induced_network"
FINDER_NAME_DCT = "name_dct"  # Dictionary of mapping of target names to reference names for species and reactions
FINDER_NUM_ASSIGNMENT_PAIR = "num_assignment_pair"
FINDER_IS_TRUNCATED = "is_truncated"
FINDER_DATAFRAME_COLUMNS = [FINDER_REFERENCE_NAME,  FINDER_TARGET_NAME,  FINDER_REFERENCE_NETWORK,
      FINDER_INDUCED_NETWORK,  FINDER_NAME_DCT, FINDER_NUM_ASSIGNMENT_PAIR, FINDER_IS_TRUNCATED]
# Data columns
D_MODEL_NAME = "model_name"
D_NUM_ITERATION = "num_iteration"
D_REFERENCE_MODEL = "reference_model"
D_TARGET_MODEL = "target_model"
D_NUM_SPECIES = "num_species"
D_NUM_REACTION = "num_reaction"
D_PROBABILITY_OF_OCCURRENCE_WEAK = "probability_of_occurrence_weak"
D_TRUNCATED_WEAK = "truncated_weak"
D_MEAN_TRUNCATED = "mean_truncated"
D_MEAN_PROBABILITY = "mean_probability"
D_STD_PROBABILITY = "std_probability"
D_PROBABILITY_OF_OCCURRENCE_STRONG = "probability_of_occurrence_strong"
D_NUM_REPLICATION = "num_replication"
D_TRUNCATED_STRONG = "truncated_strong"
D_IS_BOUNDARY_NETWORK = "is_boundary_network"
# Data paths
BIOMODELS_SERIALIZATION_PATH = os.path.join(DATA_DIR, "biomodels_serialized.txt")   # Serialized BioModels
OSCILLATORS_SERIALIZATION_PATH = os.path.join(DATA_DIR, "oscillators_serialized.txt")   # Serialized BioModels
FULL_BIOMODELS_STRONG_PATH = os.path.join(DATA_DIR, "full_biomodels_strong.csv")
FULL_BIOMODELS_WEAK_PATH = os.path.join(DATA_DIR, "full_biomodels_weak.csv")
SUBNET_BIOMODELS_STRONG_PRELIMINARY_PATH = os.path.join(AUXILIARY_DATA_DIR, "subnet_biomodels_strong_preliminary.csv")
SUBNET_BIOMODELS_WEAK_PRELIMINARY_PATH = os.path.join(AUXILIARY_DATA_DIR, "subnet_biomodels_weak_preliminary.csv")
SUBNET_BIOMODELS_STRONG_PATH = os.path.join(DATA_DIR, "subnet_biomodels_strong.csv")
SUBNET_BIOMODELS_WEAK_PATH = os.path.join(DATA_DIR, "subnet_biomodels_weak.csv")
BIOMODELS_SUMMARY_PRELIMINARY_PATH = os.path.join(AUXILIARY_DATA_DIR, "biomodels_summary_preliminary.csv")
BIOMODELS_SUMMARY_PATH = os.path.join(DATA_DIR, "biomodels_summary.csv")
