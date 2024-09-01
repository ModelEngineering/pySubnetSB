'''Find the models that have the same structure'''

import os
import pandas as pd # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore
from sirn.cluster_builder import ClusterBuilder  # type: ignore
import sirn.constants as cn  # type: ignore
import argparse

PREFIX = "identity_"
PREFIX_DCT = {True: "strong", False: "weak"}


def find_identity_collections(directory_name, is_sirn=True,
        is_strong=False, max_num_perm=cn.MAX_NUM_PERM):
    """Serializes Antimony models."""
    prefix = f"{PREFIX_DCT[is_strong]}{max_num_perm}_"
    filename = f'{directory_name}_serializers.csv'
    csv_file = os.path.join(cn.DATA_DIR, filename)
    df = pd.read_csv(csv_file)
    network_collection = NetworkCollection.deserialize(df)
    builder = ClusterBuilder(network_collection,
               identity=is_strong, is_sirn=is_sirn,
               max_num_assignment=max_num_perm)
    builder.cluster()
    output_path = os.path.join(cn.DATA_DIR, f'{prefix}{directory_name}.txt')
    with open(output_path, 'w') as f:
        for network_collection in builder.clustered_network_collections:
            f.write(str(network_collection) + '\n')


if __name__ == '__main__':
    if False:
        for directory in cn.OSCILLATOR_DIRS:
            for is_strong in [True, False]:
                for max_num_perm in [100, 1000, 10000, 100000, 1000000]:
                    print("***", directory, "***")
                    find_identity_collections(directory, is_strong=is_strong,
                        max_num_perm=max_num_perm)
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('directory_name', type=str, help='Name of directory')
    parser.add_argument('max_perm', type=int, help='Max number of permutations')
    args = parser.parse_args()
    if not args.directory_name in cn.OSCILLATOR_DIRS:
        raise ValueError(f"{args.directory_name} not in {cn.OSCILLATOR_DIRS}")
    find_identity_collections(args.directory_name, max_num_perm=args.max_perm,
                              is_sirn=False, is_strong=True)