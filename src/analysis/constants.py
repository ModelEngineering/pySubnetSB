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
if os.path.isfile(OSCILLATOR_ZIP):
    IS_OSCILLATOR_ZIP = True
else:
    IS_OSCILLATOR_ZIP = False
SIRN = "sirn"
NAIVE = "naive"

OSCILLATOR_PROJECT_DIR = "/Users/jlheller/home/Technical/repos/OscillatorDatabase"
# ResultAccessor.dataframe
# Dataframe columns
COL_HASH = "hash"
COL_OSCILLATOR_DIR = "oscillator_dir"
COL_MODEL_NAME = "model_name"
COL_PROCESSING_TIME = "processing_time"
COL_PROCESSING_TIME_MEAN = "processing_time_mean"
COL_PROCESSING_TIME_MIN = "processing_time_min"
COL_PROCESSING_TIME_MAX = "processing_time_max"
COL_PROCESSING_TIME_COUNT = "processing_time_count"
COL_PROCESSING_TIME_TOTAL = "processing_time_total"
COL_COLLECTION_IDX = "collection_idx"
COL_COLLECTION_IDX_MEAN = "collection_idx_mean"
COL_COLLECTION_IDX_MIN = "collection_idx_min"
COL_COLLECTION_IDX_MAX = "collection_idx_max"
COL_COLLECTION_IDX_COUNT = "collection_idx_count"
COL_COLLECTION_IDX_TOTAL = "collection_idx_total"
COL_IS_INDETERMINATE = "is_indeterminate"
COL_IS_INDETERMINATE_MEAN = "is_indeterminate_mean"
COL_IS_INDETERMINATE_MIN = "is_indeterminate_min"
COL_IS_INDETERMINATE_MAX = "is_indeterminate_max"
COL_IS_INDETERMINATE_COUNT = "is_indeterminate_count"
COL_IS_INDETERMINATE_TOTAL = "is_indeterminate_total"
COL_NUM_PERM = "num_perm"
COL_NUM_PERM_MEAN = "num_perm_mean"
COL_NUM_PERM_MIN = "num_perm_min"
COL_NUM_PERM_MAX = "num_perm_max"
COL_NUM_PERM_COUNT = "num_perm_count"
COL_NUM_PERM_TOTAL = "num_perm_total"
COL_CLUSTER_SIZE_EQ1_TOTAL = "cluster_size_eq1_total"
COL_CLUSTER_SIZE_EQ1_MEAN = "cluster_size_eq1_mean"
COL_CLUSTER_SIZE_EQ1_COUNT = "cluster_size_eq1_total"
COL_CLUSTER_SIZE_TOTAL = "cluster_size_total"
COL_CLUSTER_SIZE_MEAN = "cluster_size_mean"
COL_CLUSTER_SIZE_MAX = "cluster_size_MAX"
COL_CLUSTER_SIZE_COUNT = "cluster_size_count"
COL_CLUSTER_SIZE_GT1_TOTAL = "cluster_size_gt1_total"
COL_CLUSTER_SIZE_GT1_MEAN = "cluster_size_gt1_mean"
COL_CLUSTER_SIZE_GT1_MAX = "cluster_size_gt1_MAX"
COL_CLUSTER_SIZE_GT1_COUNT = "cluster_size_gt1_count"
RESULT_ACCESSOR_COLUMNS = [COL_HASH, COL_MODEL_NAME, COL_PROCESSING_TIME, COL_NUM_PERM,
    COL_IS_INDETERMINATE, COL_COLLECTION_IDX]
STATISTICS_COLUMNS = [
    COL_CLUSTER_SIZE_EQ1_TOTAL, COL_CLUSTER_SIZE_EQ1_MEAN, COL_CLUSTER_SIZE_EQ1_COUNT,
    COL_CLUSTER_SIZE_TOTAL, COL_CLUSTER_SIZE_MEAN, COL_CLUSTER_SIZE_COUNT, COL_CLUSTER_SIZE_MAX,
    COL_CLUSTER_SIZE_GT1_TOTAL, COL_CLUSTER_SIZE_GT1_MEAN, COL_CLUSTER_SIZE_GT1_COUNT, COL_CLUSTER_SIZE_GT1_MAX,
]
STATISTICS_COLUMNS.extend(RESULT_ACCESSOR_COLUMNS)
# Dataframe metadata
META_IS_STRONG = "is_strong"
META_MAX_NUM_PERM = "max_num_perm"
META_ANTIMONY_DIR = "antimony_dir"
WEAK = "weak"
STRONG = "strong"
MAX_NUM_PERMS = [100, 1000, 10000, 100000, 1000000]