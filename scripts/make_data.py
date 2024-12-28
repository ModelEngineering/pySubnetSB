"""Creates CSV files in the data directory."""

"""
Assumes
1. Files present
    cn.BIOMODELS_SERIALIZATION_PATH
    cn.OSCILOSCOPE_SERIALIZATION_PATH

To do
1. add to model summary is_boundary_model

"""
import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore

import argparse
import os
import numpy as np
import pandas as pd # type: ignore
import tqdm # type: ignore
from typing import List

MAX_SIZE_FOR_POC = 3  # Maximum number of reactions/species to calculate probability of occurrence
NUM_ITERATION_SIGNIFICANCE = 100000
NUM_ITERATION_SIGNIFICANCE_PAIRS = 10000
COLUMNS = [cn.D_MODEL_NAME, cn.D_NUM_REACTION, cn.D_NUM_SPECIES,
        cn.D_PROBABILITY_OF_OCCURRENCE_STRONG, cn.D_PROBABILITY_OF_OCCURRENCE_WEAK,
        cn.D_TRUNCATED_WEAK, cn.D_TRUNCATED_STRONG, cn.D_IS_BOUNDARY_NETWORK]

# Test paths
TEST_SUBNET_BIOMODELS_STRONG_PATH = "/tmp/subnet_biomodels_strong.csv"
TEST_SUBNET_BIOMODELS_WEAK_PATH = "/tmp/subnet_biomodels_weak.csv"
TEST_BIOMODELS_SUMMARY_PATH = "/tmp/biomodels_summary.csv"


def makeRowName(reference_name:str, target_name:str)->str:
    return reference_name + "_" + target_name

SKIP_ROWS = [makeRowName("BIOMD000000866", "BIOMD0000000942"),
            ]

def printHeader(name:str):
    print(f"\n***Processing {name}***\n")

def makeSubnetData(input_path:str, output_path:str,
      serialization_path:str=cn.BIOMODELS_SERIALIZATION_PATH):
    """
    Creates a CSV file that selects those reference, target pairs for which there is an induced network.
    Calculates the probability of occurrence if the number of reactions > 3.

    Args:
        input_path (str): Path CSV file produced by SubnetFinder
        output_path (str): Path to the output CSV file
        serialization_path (str): Path to the serialized networks
    """
    printHeader("makeSubnetData")
    serializer = ModelSerializer(None, serialization_path=serialization_path)
    networks = serializer.deserialize().networks
    network_dct = {network.network_name: network for network in networks}
    # Extract the pairs where subnets where found
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")    
    df = pd.read_csv(input_path)
    df = df[df[cn.FINDER_NUM_ASSIGNMENT_PAIR] > 0]
    # Calculate the probability of occurrence of the reference network in the target
    if os.path.isfile(output_path):
        # Read the existing data
        print(f"File exists: {output_path}. Using existing data.")
        existing_df = pd.read_csv(output_path)
        rows = [r for _, r in existing_df.iterrows()]
        processed_row_names = [makeRowName(r[cn.FINDER_REFERENCE_NAME], r[cn.FINDER_TARGET_NAME])
              for _, r in existing_df.iterrows()]
    else:
        rows = []
        processed_row_names = []
    num_rows = len(df)
    for i, row in tqdm.tqdm(df.iterrows(), desc="subnets", total=num_rows):
        # Process each network pair
        reference_network = network_dct[row[cn.FINDER_REFERENCE_NAME]]
        target_network = network_dct[row[cn.FINDER_TARGET_NAME]]
        row_name = makeRowName(row[cn.FINDER_REFERENCE_NAME], row[cn.FINDER_TARGET_NAME])
        if row_name in SKIP_ROWS:
            continue
        if row_name in processed_row_names:
            continue
        new_row = row.copy()
        # Add the probability of occurrence if the number of reactions/species is less than the threshold
        if reference_network.num_reaction <= MAX_SIZE_FOR_POC:
            new_row[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG] = np.nan
            new_row[cn.D_TRUNCATED_STRONG] = np.nan
            new_row[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK] = np.nan
            new_row[cn.D_TRUNCATED_WEAK] = np.nan
        else:
            calculator = SignificanceCalculator(reference_network, target_network.num_reaction,
                target_network.num_species, identity=cn.ID_STRONG)
            result_strong = calculator.calculate(NUM_ITERATION_SIGNIFICANCE_PAIRS, is_exact=False,
                max_num_assignment=cn.MAX_NUM_ASSIGNMENT, is_report=False)
            calculator = SignificanceCalculator(reference_network, target_network.num_reaction,
                target_network.num_species, identity=cn.ID_WEAK)
            result_weak = calculator.calculate(NUM_ITERATION_SIGNIFICANCE_PAIRS, is_exact=False,
                max_num_assignment=cn.MAX_NUM_ASSIGNMENT, is_report=False)
            new_row[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG] = result_strong.frac_induced
            new_row[cn.D_TRUNCATED_STRONG] = result_strong.frac_truncated
            new_row[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK] = result_weak.frac_induced
            new_row[cn.D_TRUNCATED_WEAK] = result_weak.frac_truncated
        processed_row_names.append(row_name)
        rows.append(new_row)
        new_df = pd.DataFrame(rows)
        new_df.to_csv(output_path, index=False)
    return df

def updateSubnetPOC(input_path:str, model_summary_path:str, output_path:str):
    """
    Updates the probability of occurrence of the reference network in the target.
    Updates are based on a lower bound approximation of POC of the reference in the target.

    Args:
        input_path (str): Path CSV file produced for makeSubnetData
        output_path (str): Path to the output CSV file
        model_summary_path (str): Path to the model summary CSV file
    """
    printHeader("updateSubnetPOC")
    subnet_df = pd.read_csv(input_path)
    summary_df = pd.read_csv(model_summary_path)
    summary_df = summary_df.set_index(cn.D_MODEL_NAME)
    # Create the POC estimates
    strong_poc_estimates = []
    weak_poc_estimates = []
    for idx, row in tqdm.tqdm(subnet_df.iterrows(), desc="subnets", total=len(subnet_df)):
        reference_name = row[cn.FINDER_REFERENCE_NAME]
        target_name = row[cn.FINDER_TARGET_NAME]
        num_reference_reaction = summary_df.loc[reference_name][cn.D_NUM_REACTION]
        subnet_strong_poc = row[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG]
        subnet_weak_poc = row[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]
        if num_reference_reaction > MAX_SIZE_FOR_POC:
            continue
        if not np.isnan(subnet_strong_poc):
            strong_poc_estimates.append(subnet_strong_poc)
        if not np.isnan(subnet_weak_poc):
            weak_poc_estimates.append(subnet_weak_poc)
        reference_strong_poc = summary_df[reference_name][cn.D_PROBABILITY_OF_OCCURRENCE_STRONG]
        reference_weak_poc = summary_df[reference_name][cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]
        num_target_reaction = summary_df.loc[target_name][cn.D_NUM_REACTION]
        target_reference_ratio = num_target_reaction / num_reference_reaction
        estimate_strong_poc = 1 - (1 - reference_strong_poc) ** target_reference_ratio
        estimate_weak_poc = 1 - (1 - reference_weak_poc) ** target_reference_ratio
        strong_poc_estimates.append(estimate_strong_poc)
        weak_poc_estimates.append(estimate_weak_poc)
    # Update the dataframe
    subnet_df[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG] = strong_poc_estimates
    subnet_df[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK] = weak_poc_estimates
    subnet_df.to_csv(output_path, index=False)

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
    printHeader("makeModelSummary")
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
    for idx, network in tqdm.tqdm(enumerate(networks), desc="networks", total=len(networks)):
        if idx % num_worker != worker_idx:
            continue
        if network.network_name in processed_network_names:
            continue
        dct[cn.D_MODEL_NAME].append(network.network_name)
        dct[cn.D_NUM_REACTION].append(network.num_reaction)
        dct[cn.D_NUM_SPECIES].append(network.num_species)
        if network.num_reaction <= num_reaction_threshold:
            # Calculate probability of occurrence
            try:
                result_strong = SignificanceCalculator.calculateNetworkOccurrenceProbability(network,
                    num_iteration=num_iteration, is_report=False,
                    identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT)
                if result_strong[1] > 0.001:
                    print(f"Truncation is too high for strong POC: {result_strong[1]} for {network.network_name}")  
                #
                result_weak = SignificanceCalculator.calculateNetworkOccurrenceProbability(network,
                    num_iteration=num_iteration, is_report=False,
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
        dct[cn.D_TRUNCATED_WEAK].append(result_weak[1])
        dct[cn.D_TRUNCATED_STRONG].append(result_strong[1])
        dct[cn.D_IS_BOUNDARY_NETWORK].append(network.isBoundaryNetwork())
        df = pd.DataFrame(dct)
        df.to_csv(worker_output_path, index=False)

def mergeBiomodelsSummary(dfs:List[pd.DataFrame],
      serialization_path:str=cn.BIOMODELS_SERIALIZATION_PATH)->pd.DataFrame:
    """
    Merges multiple model by calculating the median of the probability of occurrence.
    Also, add is_boundary_network if it's missing.
    """
    printHeader("mergeModelBiomodelsSummary")
    result_df = dfs[0].copy().set_index(cn.D_MODEL_NAME)
    # Get the networks
    serializer = ModelSerializer(None, serialization_path=serialization_path)
    networks = serializer.deserialize().networks
    #
    merged_df = pd.concat(dfs)
    group_dct = merged_df.groupby(cn.D_MODEL_NAME).groups
    for model_name, idx_lst in group_dct.items():
        idxs = idx_lst.to_list()
        for column in [cn.D_PROBABILITY_OF_OCCURRENCE_STRONG, cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]:
            values = [dfs[idx].loc[idxs[idx], column]
              for idx in range(len(idx_lst))]
            pruned_values = [v for v in values if not np.isnan(v)]
            if len(pruned_values) == 0:
                median = np.nan
            else:
                median = np.median(pruned_values)
            result_df.loc[model_name, column] = median
    # Augment with is_boundary_network
    if not cn.D_IS_BOUNDARY_NETWORK in result_df.columns:
        for network in networks:
            if network.network_name not in result_df.index:
                continue
            result_df.loc[network.network_name, cn.D_IS_BOUNDARY_NETWORK] = network.isBoundaryNetwork()
    # Return result
    result_df = result_df.reset_index()


if __name__ == '__main__':
    subnet_biomodels_strong_path = cn.SUBNET_BIOMODELS_STRONG_PATH
    subnet_biomodels_weak_path = cn.SUBNET_BIOMODELS_WEAK_PATH
    biomodels_summary_path = cn.BIOMODELS_SUMMARY_PATH
    is_makeSubnetData = False
    is_makeModelSummary = False
    is_mergeBiomodelsSummary = False
    #
    if is_makeSubnetData:
        makeSubnetData(cn.FULL_BIOMODELS_STRONG_PATH, subnet_biomodels_strong_path)
        makeSubnetData(cn.FULL_BIOMODELS_WEAK_PATH, subnet_biomodels_weak_path)
    if is_makeModelSummary:
        makeModelSummary(cn.BIOMODELS_SERIALIZATION_PATH, biomodels_summary_path,
            num_worker=1, worker_idx=0)
    if 
        df_paths = [os.path.join(cn.DATA_DIR, f) for f in os.listdir(cn.DATA_DIR) if f.endswith("csv") and "biomodels_summary_" in f]  
        dfs = [pd.read_csv(p) for p in df_paths]
        mergeBiomodelsSummary(dfs)