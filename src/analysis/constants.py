import sirn.constants as cnn # type: ignore
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TEST_DIR = os.path.join(PROJECT_DIR, 'analysis_tests')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SIRN_DIR = os.path.join(DATA_DIR, "sirn_analysis")
NAIVE_DIR = os.path.join(DATA_DIR, "naive_analysis")
OSCILLATOR_ZIP = os.path.join(DATA_DIR, "oscillators.zip")

OSCILLATOR_PROJECT_DIR = "/Users/jlheller/home/Technical/repos/OscillatorDatabase"
# ResultAccessor.dataframe
# Dataframe columns
COL_HASH = "hash"
COL_ANTIMONY_DIR = "antimony_dir"
COL_MODEL_NAME = "model_name"
COL_PROCESSING_TIME = "processing_time"
COL_NUM_PERM = "num_perm"
COL_IS_INDETERMINATE = "is_indeterminate"
COL_COLLECTION_IDX = "collection_idx"
COL_CLUSTERGT1 = "cluster_gt1"
RESULT_ACCESSOR_COLUMNS = [COL_HASH, COL_MODEL_NAME, COL_PROCESSING_TIME, COL_NUM_PERM,
           COL_IS_INDETERMINATE, COL_COLLECTION_IDX]
# Dataframe metadata
META_IS_STRONG = "is_strong"
META_MAX_NUM_PERM = "max_num_perm"
META_ANTIMONY_DIR = "antimony_dir"
WEAK = "weak"
STRONG = "strong"
MAX_NUM_PERMS = [100, 1000, 10000, 100000, 1000000]
# Metrics
M_NUM_MODEL = "num_model"
M_NUM_PERM = "num_perm"
M_INDETERMINATE = "indeterminate"
M_PROCESSING_TIME = "processing_time"
M_CLUSTER_SIZE = "cluster_size"
M_CLUSTER_SIZE_EQ1 = "cluster_size_eq1"
M_CLUSTER_SIZE_GT1 = "cluster_size_gt1"
METRICS = [M_NUM_MODEL, M_NUM_PERM, M_INDETERMINATE, M_PROCESSING_TIME, M_CLUSTER_SIZE,
           M_CLUSTER_SIZE_EQ1, M_CLUSTER_SIZE_GT1]