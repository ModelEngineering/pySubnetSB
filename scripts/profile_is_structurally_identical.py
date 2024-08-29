import sirn.constants as cn  # type: ignore
from sirn.network import Network # type: ignore

import numpy as np


IS_REPORT = True


def profileIsStructurallyIdentical(reference_size, target_factor=1, max_num_assignment=100000000):
    num_iteration = 100
    success_cnt = 0
    total_cnt = 0
    for _ in range(num_iteration):
        for identity in [cn.ID_WEAK, cn.ID_STRONG]:
            for is_subsets in [True, False]:
                if (not is_subsets) and (target_factor > 1):
                    continue
                reference = Network.makeRandomNetworkByReactionType(reference_size)
                target, assignment_pair = reference.permute()
                target_reactant_arr = np.hstack([target.reactant_mat.values]*target_factor)
                target_product_arr = np.hstack([target.product_mat.values]*target_factor)
                target = Network(target_reactant_arr, target_product_arr)
                if True:
                    result = reference.isStructurallyIdentical(target, identity=identity, is_subsets=is_subsets,
                            expected_assignment_pair=assignment_pair, max_num_assignment=max_num_assignment)
                    total_cnt += 1
                    if result.is_truncated:
                        continue
                    success_cnt += 1
    if IS_REPORT:
        print(f"Total count: {total_cnt}; Success count: {success_cnt}")
        print(f"  reference_size: {reference_size}; target_factor: {target_factor}")
        print(f"  max_num_assignment: {max_num_assignment}")

if __name__ == '__main__':
    profileIsStructurallyIdentical(6, target_factor=10)