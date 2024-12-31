"""Creates CSV files in the data directory."""

"""
Data produced:
1. Subnets found. cn.BIOMODELS_SUBNET_STRONG_PATH, cn.BIOMODELS_SUBNET_WEAK_PATH
2. Model summary. cn.BIOMODELS_SUMMARY_PATH

Data required:
    cn.BIOMODELS_SERIALIZATION_PATH
    cn.OSCILOSCOPE_SERIALIZATION_PATH
"""


import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.significance_calculator import SignificanceCalculator  # type: ignore

import argparse
import os
import numpy as np
import pandas as pd # type: ignore
import tqdm # type: ignore
from typing import List, Optional

MAX_SIZE_FOR_POC = 3  # Maximum number of reactions/species to calculate probability of occurrence
NUM_ITERATION_SIGNIFICANCE = 100000
NUM_ITERATION_SIGNIFICANCE_PAIRS = 10000
COLUMNS = [cn.D_MODEL_NAME, cn.D_NUM_REACTION, cn.D_NUM_SPECIES,
        cn.D_PROBABILITY_OF_OCCURRENCE_STRONG, cn.D_PROBABILITY_OF_OCCURRENCE_WEAK,
        cn.D_TRUNCATED_WEAK, cn.D_TRUNCATED_STRONG, cn.D_IS_BOUNDARY_NETWORK]

###############################################################
def makeRowName(reference_name:str, target_name:str)->str:
    return reference_name + "_" + target_name

SUBNET_BIOMODELS_SKIPS = [makeRowName("BIOMD000000866", "BIOMD0000000942"),
            ]

def _print(msg:str, is_report:bool=True):
    if is_report:
        print(msg)

def _printHeader(name:str, is_report:bool=True):
    _print(f"\n***Processing {name}***\n", is_report=is_report)

def makeSubnetData(input_path:str, output_path:str,
      serialization_path:str=cn.BIOMODELS_SERIALIZATION_PATH, num_row:int=-1,
      is_report:bool=True, subnet_skips:List[str]=SUBNET_BIOMODELS_SKIPS)->pd.DataFrame:
    """
    Creates a CSV file that selects those reference, target pairs for which there is an induced network.
    Calculates the probability of occurrence if the number of reactions > 3.

    Args:
        input_path (str): Path CSV file produced by SubnetFinder
        output_path (str): Path to the output CSV file
        serialization_path (str): Path to the serialized networks
        num_row (int): Number of rows to process (all if < 0)
        subnet_skips (List[str]): List of row names (<reference_name>_<target_name>) to skip
    """
    _printHeader("makeSubnetData", is_report=is_report)
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
        _print(f"File exists: {output_path}. Using existing data.", is_report=is_report)
        existing_df = pd.read_csv(output_path)
        rows = [r for _, r in existing_df.iterrows()]
        processed_row_names = [makeRowName(r[cn.FINDER_REFERENCE_NAME], r[cn.FINDER_TARGET_NAME])
              for _, r in existing_df.iterrows()]
    else:
        rows = []
        processed_row_names = []
    total_row = len(df)
    num_processed_row = 0
    for i, row in tqdm.tqdm(df.iterrows(), desc="subnets", total=total_row, disable=not is_report):
        # Process each network pair
        reference_network = network_dct[row[cn.FINDER_REFERENCE_NAME]]
        target_network = network_dct[row[cn.FINDER_TARGET_NAME]]
        row_name = makeRowName(row[cn.FINDER_REFERENCE_NAME], row[cn.FINDER_TARGET_NAME])
        if row_name in subnet_skips:
            continue
        if row_name in processed_row_names:
            continue
        if num_row > 0 and (num_processed_row >= num_row):
            break
        num_processed_row += 1
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

def updateSubnetPOC(input_path:str, model_summary_path:str, output_path:str, is_report:bool=True,
      num_row:int=-1):
    """
    Updates the probability of occurrence of the reference network in the target.
    Updates are based on a lower bound approximation of POC of the reference in the target.

    Args:
        input_path (str): Path CSV file produced for makeSubnetData
        output_path (str): Path to the output CSV file
        model_summary_path (str): Path to the model summary CSV file
    """
    _printHeader("updateSubnetPOC", is_report=is_report)
    subnet_df = pd.read_csv(input_path)
    summary_df = pd.read_csv(model_summary_path)
    summary_df = summary_df.set_index(cn.D_MODEL_NAME)
    # Create the POC estimates
    strong_poc_estimates = []
    weak_poc_estimates = []
    num_processed_row = 0
    for _, row in tqdm.tqdm(subnet_df.iterrows(), desc="subnets", total=len(subnet_df)):
        reference_name = row[cn.FINDER_REFERENCE_NAME]
        target_name = row[cn.FINDER_TARGET_NAME]
        num_reference_reaction = summary_df.loc[reference_name][cn.D_NUM_REACTION]
        subnet_strong_poc = row[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG]
        subnet_weak_poc = row[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]
        if num_row > 0 and (num_processed_row >= num_row):
            break
        num_processed_row += 1
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

def makeModelSummary(
      input_path:str=cn.BIOMODELS_SERIALIZATION_PATH,
      output_path:str=cn.BIOMODELS_SUMMARY_PATH,
      num_reaction_threshold:int=10,
      num_iteration:int=NUM_ITERATION_SIGNIFICANCE,
      num_worker:int=1,
      worker_idx:int=0,
      is_report:bool=True,
      total_model:int=-1):
    """
    Creates a CSV file that summarizes the models. This is parallelized into many workers.
    The output file is: <output_path>_worker_<worker_idx>.csv
    Only includes models with at least one reaction.
    Since the probability of occurrence has some variability, this process may be done multiple times.

    Args:
        input_path (str): Path to the serialized networks
        output_path (str): Path to the output CSV file
        num_reaction_threshold (int): Maximum number of reactions in model to calculate probability of occurrence
        num_iteration (int): Number of iterations
        num_worker (int): Number of workers
        worker_idx (int): Worker index
        total_model (int): Total number of models to process
    """
    _printHeader("makeModelSummary", is_report=is_report)
    # Initialize
    worker_output_path = output_path.replace(".csv", f"_worker_{worker_idx}.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")    
    serializer = ModelSerializer(None, serialization_path=input_path)
    networks = serializer.deserialize().networks
    # Construct the results dictionary
    dct:dict = {c: [] for c in COLUMNS}
    processed_network_names:list = []
    if os.path.isfile(worker_output_path):
        _print(f"File exists: {worker_output_path}. Using existing data.", is_report=is_report)
        df = pd.read_csv(worker_output_path)
        for c in COLUMNS:
            dct[c] = df[c].tolist()
        processed_network_names = df[cn.D_MODEL_NAME].tolist()
    # Process the networks
    networks = sorted(networks, key=lambda x: x.num_reaction)
    for idx, network in tqdm.tqdm(enumerate(networks), desc="networks", total=len(networks),
          disable=not is_report):
        if idx % num_worker != worker_idx:
            continue
        if network.network_name in processed_network_names:
            continue
        if total_model > 0 and (idx >= total_model):
            break
        dct[cn.D_MODEL_NAME].append(network.network_name)
        dct[cn.D_NUM_REACTION].append(network.num_reaction)
        dct[cn.D_NUM_SPECIES].append(network.num_species)
        if (network.num_reaction <= num_reaction_threshold) and (network.num_reaction > 0):
            # Calculate probability of occurrence
            try:
                result_strong = SignificanceCalculator.calculateNetworkOccurrenceProbability(network,
                    num_iteration=num_iteration, is_report=False,
                    identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT)
                if result_strong[1] > 0.001:
                    _print(f"Truncation is too high for strong POC: {result_strong[1]} for {network.network_name}",
                          is_report=is_report)  
                #
                result_weak = SignificanceCalculator.calculateNetworkOccurrenceProbability(network,
                    num_iteration=num_iteration, is_report=False,
                    identity=cn.ID_STRONG, max_num_assignment=cn.MAX_NUM_ASSIGNMENT)
                if result_weak[1] > 0.001:
                    _print(f"Truncation is too high for weak POC: {result_weak[1]} for {network.network_name}",
                           is_report=is_report)  
            except Exception as e:
                _print(f"Error in {network.network_name} calculating probability of occurrence: {e}",
                      is_report=is_report)
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

def consolidateBiomodelsSummary(input_path:str=cn.BIOMODELS_SUMMARY_MULTIPLE_PATH,
      output_path:str=cn.BIOMODELS_SUMMARY_PATH, is_report:bool=True):
    """
    Calculates the median probability of occurrence for each model in the input files.
    Models may have multiple occurrences. The output has only one occurrence of each model.
    Add is_boundary_network if it's missing.
    """
    _printHeader("mergeModelBiomodelsSummary", is_report=is_report)
    df = pd.read_csv(input_path)
    # Find the median values
    sub1_df = df[[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG, cn.D_PROBABILITY_OF_OCCURRENCE_WEAK,
          cn.D_MODEL_NAME]]
    median_df = sub1_df.groupby(cn.D_MODEL_NAME).median()
    # Create dataframe for constant columns
    sub2_df = df.copy()
    del sub2_df[cn.D_PROBABILITY_OF_OCCURRENCE_STRONG]
    del sub2_df[cn.D_PROBABILITY_OF_OCCURRENCE_WEAK]
    sub2_df = sub2_df.drop_duplicates()
    #
    result_df = sub2_df.merge(median_df, left_on=cn.D_MODEL_NAME, right_index=True)
    result_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Function invocations are ordered by data depenencies
    is_make_subnet_data = False
    is_make_model_summary = False
    is_consolidate_biomodels_summary = False
    is_update_subnet_POC = False
    #
    if is_make_subnet_data:
        # Find models in BioModels that are subnets of other models
        makeSubnetData(cn.FULL_BIOMODELS_STRONG_PATH, cn.SUBNET_BIOMODELS_STRONG_PATH)
        makeSubnetData(cn.FULL_BIOMODELS_WEAK_PATH, cn.SUBNET_BIOMODELS_WEAK_PATH)
    if is_make_model_summary:
        # Create summary statistics for each model
        parser = argparse.ArgumentParser(description='Make Model Summary')
        parser.add_argument('num_worker', type=int, help='Number of workers')
        parser.add_argument('worker_idx', type=int, help='Worker index')
        args = parser.parse_args()
        makeModelSummary(cn.BIOMODELS_SERIALIZATION_PATH, cn.BIOMODELS_SUMMARY_PATH,
            num_worker=args.num_worker, worker_idx=args.worker_idx)
    if is_consolidate_biomodels_summary:
        # Merge multiple summaries are produced (because of replications of simulations)
        df_paths = [os.path.join(cn.DATA_DIR, f) for f in os.listdir(cn.DATA_DIR)
              if f.endswith("csv") and "biomodels_summary_" in f]  
        dfs = [pd.read_csv(p) for p in df_paths]
        consolidateBiomodelsSummary(input_path=cn.BIOMODELS_SUMMARY_MULTIPLE_PATH,
              output_path=cn.BIOMODELS_SUMMARY_PATH)
    if is_update_subnet_POC:
        # Update the probability of occurrence of induced networks for small models using
        #   an analytic approximation
        updateSubnetPOC(cn.SUBNET_BIOMODELS_STRONG_PATH, cn.BIOMODELS_SUMMARY_PATH,
              cn.SUBNET_BIOMODELS_STRONG_AUGMENTED_PATH)
        updateSubnetPOC(cn.SUBNET_BIOMODELS_WEAK_PATH, cn.BIOMODELS_SUMMARY_PATH,
              cn.SUBNET_BIOMODELS_WEAK_AUGMENTED_PATH)