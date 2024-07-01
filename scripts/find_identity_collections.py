'''Find the models that have the same structure'''

import os
import pandas as pd # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore
from sirn.cluster_builder import ClusterBuilder  # type: ignore
import sirn.constants as cn  # type: ignore
import argparse

PREFIX = "identity_"
PREFIX_DCT = {True: "strong", False: "weak"}


def find_identity_collections(directory_name,
        is_strong=False, max_num_perm=cn.MAX_NUM_PERM):
    """Serializes Antimony models."""
    prefix = f"{PREFIX_DCT[is_strong]}{max_num_perm}_"
    filename = f'{directory_name}_serializers.csv'
    csv_file = os.path.join(cn.DATA_DIR, filename)
    df = pd.read_csv(csv_file)
    network_collection = NetworkCollection.deserialize(df)
    builder = ClusterBuilder(network_collection,
               is_structural_identity_strong=is_strong,
               max_num_perm=max_num_perm)
    builder.cluster()
    output_path = os.path.join(cn.DATA_DIR, f'{prefix}{directory_name}.txt')
    with open(output_path, 'w') as f:
        for network_collection in builder.clustered_network_collections:
            f.write(str(network_collection) + '\n')


if __name__ == '__main__':
    for directory in cn.OSCILLATOR_DIRS:
        for is_strong in [True, False]:
            #for max_num_perm in [100, 1000, 10000, 100000]:
            for max_num_perm in [100, 100000]:
                print("***", directory, "***")
                find_identity_collections(directory, is_strong=is_strong,
                     max_num_perm=max_num_perm)