'''Validation of measurements and data'''

import os
import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.result_accessor import ResultAccessor  # type: ignore

PREFIX = "identity_"
PREFIX_DCT = {True: "strong", False: "weak"}

def checkFiles(dir_path:str):
    """Check that all subdirectories have all oscillator processed files."""
    for subdir in os.listdir(dir_path):
        print(f"***Processing {dir_path}/{subdir}***")
        split = subdir.split(".")
        if len(split) > 1:
            continue
        iter = ResultAccessor.iterateDir(os.path.join(dir_path, subdir))
        present_files = [f for f, _ in list(iter)]
        for expected_file in cnn.OSCILLATOR_DIRS:
            expected_presents = [expected_file in f for f in present_files]
            if sum(expected_presents) == 0:
                print(f"  Missing {expected_file}")


def validateSIRNvsNaive(is_strong=True):
    """
    Validate that the naive analysis is a subset of the SIRN analysis.

    Args:
        is_strong: True for strong, False for weak
    """
    PREFIX_DCT = {True: "strong", False: "weak"}
    for max_num_perm in cn.MAX_NUM_PERMS:
        directory = f"{PREFIX_DCT[is_strong]}{max_num_perm}"
        print(f"***{directory}***")
        subset_dir = os.path.join(cn.DATA_DIR, "naive_analysis", directory)
        superset_dir = os.path.join(cn.DATA_DIR, "sirn_analysis", directory)
        missing_dct = ResultAccessor.isClusterSubset(superset_dir, subset_dir)
        if len(missing_dct[cn.COL_ANTIMONY_DIR]) > 0:
            print(missing_dct)


if __name__ == '__main__':
    #checkFiles(os.path.join(cn.DATA_DIR, "naive_analysis"))
    #checkFiles(os.path.join(cn.DATA_DIR, "sirn_analysis"))
    validateSIRNvsNaive(is_strong=True)
    #validateSIRNvsNaive(is_strong=False)