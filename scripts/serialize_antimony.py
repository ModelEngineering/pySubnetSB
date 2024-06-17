'''Serializes Antimony Models Collected for Oscillators.'''

import os
from sirn.pmc_serializer import PMCSerializer  # type: ignore
import sirn.constants as cn  # type: ignore

OSCILLATOR_PROJECT = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    OSCILLATOR_PROJECT = os.path.dirname(OSCILLATOR_PROJECT)
OSCILLATOR_PROJECT = os.path.join(OSCILLATOR_PROJECT, 'OscillatorDatabase')
DIRS = ["Oscillators_June_10_B_10507", "Oscillators_June_9_2024_18070",
        "Oscillators_June_10_A_13440", "Oscillators_June_11_13402", "Oscillators_May_28_2024_11569" ]

def serialize_antimony(directory_name):
    """Serializes Antimony models."""
    directory_path = os.path.join(OSCILLATOR_PROJECT, directory_name)
    pmatrix_collection = PMCSerializer.makePMCollectionAntimonyDirectory(directory_path,
                                                                         report_interval=100)
    serializer = PMCSerializer(pmatrix_collection)
    df = serializer.serialize()
    output_path = os.path.join(cn.DATA_DIR, f'{directory_name}_serializers.csv')
    df.to_csv(output_path, index=False)

for dir_name in DIRS:
    print("***", dir_name, "***")
    serialize_antimony(dir_name)