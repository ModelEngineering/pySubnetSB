'''Serializes Antimony Models Collected for Oscillators.'''

import os
import pandas as pd # type: ignore
from sirn.pmc_serializer import PMCSerializer  # type: ignore
import sirn.constants as cn  # type: ignore
import argparse

OSCILLATOR_PROJECT = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    OSCILLATOR_PROJECT = os.path.dirname(OSCILLATOR_PROJECT)
OSCILLATOR_PROJECT = os.path.join(OSCILLATOR_PROJECT, 'OscillatorDatabase')


def serialize_antimony(directory_name):
    """Serializes Antimony models."""
    directory_path = os.path.join(OSCILLATOR_PROJECT, directory_name)
    output_path = os.path.join(cn.DATA_DIR, f'{directory_name}_serializers.csv')
    # Check if there is an existing output file
    if os.path.exists(output_path):
        initial_df = pd.read_csv(output_path)
        processed_model_names = list(initial_df['model_name'])
    else:
        processed_model_names = []
    pmatrix_collection = PMCSerializer.makePMCollectionAntimonyDirectory(directory_path,
            processed_model_names=processed_model_names, report_interval=100)
    serializer = PMCSerializer(pmatrix_collection)
    df = serializer.serialize()
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('directory_name', type=str, help='Name of directory')
    args = parser.parse_args()
    if not args.directory_name in cn.OSCILLATOR_DIRS:
        raise ValueError(f"{args.directory_name} not in {cn.OSCILLATOR_DIRS}")
    serialize_antimony(args.directory_name)