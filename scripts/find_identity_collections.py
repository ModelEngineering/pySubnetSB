'''Find the models that have the same structure'''

import os
import pandas as pd # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore
from sirn.cluster_builder import ClusterBuilder  # type: ignore
import sirn.constants as cn  # type: ignore
import argparse

PREFIX = "identity_"
PREFIX_DCT = {cn.ID_STRONG: "strong", cn.ID_WEAK: "weak"}


def find_identity_collections(directory_name,
        identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT):
    """Serializes Antimony models."""
    prefix = f"{PREFIX_DCT[identity]}{max_num_assignment}_"
    filename = f'{directory_name}_serializers.csv'
    csv_file = os.path.join(cn.DATA_DIR, filename)
    df = pd.read_csv(csv_file)
    df = df.rename(columns={'num_col': cn.S_NUM_REACTION, 'num_row': cn.S_NUM_SPECIES,
          'column_names': cn.S_REACTION_NAMES, 'row_names': cn.S_SPECIES_NAMES,
          'reactant_array_str': cn.S_REACTANT_LST, 'product_array_str': cn.S_PRODUCT_LST,
          'model_name': cn.S_NETWORK_NAME})
    serialization_str = NetworkCollection.dataframeToJson(df)
    network_collection = NetworkCollection.deserialize(serialization_str)
    builder = ClusterBuilder(network_collection,
               identity=identity,
               max_num_assignment=max_num_assignment)
    builder.cluster()
    output_path = os.path.join(cn.DATA_DIR, f'{prefix}{directory_name}.txt')
    with open(output_path, 'w') as f:
        for processed_network_collection in builder.processed_network_collections:
            f.write(processed_network_collection.serialize() + '\n')
            #f.write(str(processed_network_collection) + '\n')


if __name__ == '__main__':
    if False:
        for directory in cn.OSCILLATOR_DIRS:
            for is_strong in [True, False]:
                for max_num_assignment in [100, 1000, 10000, 100000, 1000000]:
                    print("***", directory, "***")
                    find_identity_collections(directory, is_strong=is_strong,
                        max_num_assignment=max_num_assignment)
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('directory_name', type=str, help='Name of directory')
    parser.add_argument('max_num_assignment', type=int, help='Max number of permutations')
    args = parser.parse_args()
    if not args.directory_name in cn.OSCILLATOR_DIRS:
        raise ValueError(f"{args.directory_name} not in {cn.OSCILLATOR_DIRS}")
    find_identity_collections(args.directory_name, max_num_assignment=args.max_num_assignment,
                              identity=cn.ID_STRONG)