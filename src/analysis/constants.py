import sirn.constants as cnn # type: ignore
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TEST_DIR = os.path.join(PROJECT_DIR, 'analysis_tests')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

OSCILLATOR_PROJECT_DIR = "/Users/jlheller/home/Technical/repos/OscillatorDatabase"
# ResultAccessor.dataframe
# Dataframe columns
COL_HASH = "hash"
COL_MODEL_NAME = "model_name"
COL_PROCESS_TIME = "process_time"
COL_NUM_PERM = "num_perm"
COL_IS_INDETERMINATE = "is_indeterminate"
COL_COLLECTION_IDX = "collection_idx"
RESULT_ACCESSOR_COLUMNS = [COL_HASH, COL_MODEL_NAME, COL_PROCESS_TIME, COL_NUM_PERM,
           COL_IS_INDETERMINATE, COL_COLLECTION_IDX]
# Dataframe metadata
META_IS_STRONG = "is_strong"
META_MAX_NUM_PERM = "max_num_perm"
META_ANTIMONY_DIR = "antimony_dir"