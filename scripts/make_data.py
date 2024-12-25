"""Creates CSV files in the data directory."""

"""
Assumes

    Returns:
        _type_: _description_
"""
import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore

import argparse
import os
import numpy as np
import pandas as pd # type: ignore

NUM_ITERATION_SIGNIFICANCE = 100000
COLUMNS = [cn.D_MODEL_NAME, cn.D_NUM_REACTION, cn.D_NUM_SPECIES,
        cn.D_PROBABILITY_OF_OCCURRENCE_STRONG, cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]


def makeSubnetData(input_path:str, output_path:str):
    """
    Creates a CSV file that selects those reference, target pairs for which there is an induced network.

    Args:
        input_path (str): Path CSV file produced by SubnetFinder
        output_path (str): Path to the output CSV file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")    
    df = pd.read_csv(input_path)
    df = df[df[cn.FINDER_NUM_ASSIGNMENT_PAIR] > 0]
    df.to_csv(output_path, index=False)

def makeModelSummary(input_path:str, output_path:str, num_reaction_threshold:int=10,
      num_iteration:int=NUM_ITERATION_SIGNIFICANCE, num_worker:int=1, worker_idx:int=0):
    """
    Creates a CSV file that summarizes the models.

    Args:
        input_path (str): Path to the serialized networks
        output_path (str): Path to the output CSV file
        num_reaction_threshold (int): Maximum number of reactions in model to calculate probability of occurrence
        num_iteration (int): Number of iterations
        num_worker (int): Number of workers
        worker_idx (int): Worker index
    """
    worker_output_path = output_path.replace(".csv", f"_worker_{worker_idx}.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")    
    serializer = ModelSerializer(None, serialization_path=input_path)
    networks = serializer.deserialize().networks
    # Construct the results dictionary
    dct:dict = {c: [] for c in COLUMNS}
    processed_network_names:list = []
    if os.path.isfile(worker_output_path):
        print(f"File exists: {worker_output_path}. Using existing data.")
        df = pd.read_csv(worker_output_path)
        for c in COLUMNS:
            dct[c] = df[c].tolist()
        processed_network_names = df[cn.D_MODEL_NAME].tolist()
    # Process the networks
    for idx, network in enumerate(networks):
        if idx % num_worker != worker_idx:
            continue
        if network.network_name in processed_network_names:
            continue
        print(f"Processing {network.network_name} ({idx+1}/{len(networks)})")
        dct[cn.D_MODEL_NAME].append(network.network_name)
        dct[cn.D_NUM_REACTION].append(network.num_reaction)
        dct[cn.D_NUM_SPECIES].append(network.num_species)
        if network.num_reaction <= num_reaction_threshold:
            # Calculate probability of occurrence
            try:
                result_strong = SignificanceCalculator.calculateOccurrenceProbability(network,
                    num_iteration=num_iteration,
                    identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT)
                if result_strong[1] > 0.001:
                    print(f"Truncation is too high for strong POC: {result_strong[1]} for {network.network_name}")  
                #
                result_weak = SignificanceCalculator.calculateOccurrenceProbability(network,
                    num_iteration=num_iteration,
                    identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT)
                if result_weak[1] > 0.001:
                    print(f"Truncation is too high for weak POC: {result_weak[1]} for {network.network_name}")  
            except Exception as e:
                print(f"Error in {network.network_name} calculating probability of occurrence: {e}")
                result_strong = (np.nan, np.nan)
                result_weak = (np.nan, np.nan)
        else:
            result_strong = (np.nan, np.nan)
            result_weak = (np.nan, np.nan)
        dct[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG].append(result_strong[0])
        dct[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK].append(result_weak[0])
        df = pd.DataFrame(dct)
        df.to_csv(worker_output_path, index=False)


if __name__ == '__main__':
    #makeSubnetData(cn.FINDER_DATAFRAME_PATH, cn.SUBNET_DATA_PATH)
    parser = argparse.ArgumentParser(description='Make data for pySubnetSB')
    parser.add_argument('num_worker', type=int, help='Number of workers')
    parser.add_argument('worker_idx', type=int, help='Worker index')
    args = parser.parse_args()
    makeModelSummary(cn.BIOMODELS_SERIALIZATION_PATH, cn.BIOMODELS_SUMMARY_PATH,
          num_worker=args.num_worker, worker_idx=args.worker_idx)