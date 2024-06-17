'''Find the models that have the same structure'''

import os
import pandas as pd # type: ignore
from sirn.pmc_serializer import PMCSerializer  # type: ignore
import sirn.constants as cn  # type: ignore
import argparse

PREFIX = "identity_"


def find_identity_collections(directory_name):
    """Serializes Antimony models."""
    filename = f'{directory_name}_serializers.csv'
    csv_file = os.path.join(cn.DATA_DIR, filename)
    df = pd.read_csv(csv_file)
    pmatrix_collection = PMCSerializer.deserialize(df)
    pmatrix_identity_collections = pmatrix_collection.cluster()
    output_path = os.path.join(cn.DATA_DIR, f'{PREFIX}{directory_name}.txt')
    with open(output_path, 'w') as f:
        for pmatrix_identity_collection in pmatrix_identity_collections:
            f.write(str(pmatrix_identity_collection) + '\n')


if __name__ == '__main__':
    for directory in cn.OSCILLATOR_DIRS:
        print("***", directory, "***")
        find_identity_collections(directory)
    if False:
        parser = argparse.ArgumentParser(description='Serialize Antimony Models')
        parser.add_argument('directory_name', type=str, help='Name of directory')
        args = parser.parse_args()
        find_identity_collections(args.directory_name)