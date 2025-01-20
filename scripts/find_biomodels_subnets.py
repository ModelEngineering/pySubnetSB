"""Analyzes biomodels for subnets."""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.parallel_subnet_finder import ParallelSubnetFinder  # type: ignore
from pySubnetSB.subnet_finder import SubnetFinder, NetworkPair  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore

import os
import numpy as np
import pandas as pd # type: ignore
from multiprocessing import freeze_support
from typing import List


MAX_NUM_ASSIGNMENT_INITIAL = 1000
MAX_NUM_ASSIGNMENT_FINAL = np.int64(1e12)
SKIP_NETWORKS = ["BIOMD0000000192", "BIOMD0000000394", "BIOMD0000000433", "BIOMD0000000442","BIOMD0000000432", "BIOMD0000000441", "BIOMD0000000440",
      "BIOMD0000000668", "BIOMD0000000690", "BIOMD0000000689", "BIOMD0000000038", "BIOMD0000000639", # processed killed
      "BIOMD0000000084", "BIOMD0000000296", "BIOMD0000000719",  "BIOMD0000000915",  "BIOMD0000001015", "BIOMD0000001009", "BIOMD0000000464", 
      "BIOMD0000000464",  "BIOMD0000000979", "BIOMD0000000980", # Large number of assignments, none of which are matches
      "BIOMD0000000987", 
        "BIOMD0000000284", "BIOMD0000000080",  # killed in 2nd stage
              ]
OUTPUT_CSV = "biomodels_subnet_final.csv"

def main(is_initialize:bool=False, identity:str=cn.ID_STRONG)->None:
    # Find subnets of the Biomodels database
    initial_df = ParallelSubnetFinder.biomodelsFind(
        reference_network_size=10,
        identity=identity,
        is_report=True,
        serialization_dir=cn.DATA_DIR,
        checkpoint_path=None,
        is_no_boundary_network=True,
        num_worker=-1,
        skip_networks=SKIP_NETWORKS,
        max_num_assignment=MAX_NUM_ASSIGNMENT_INITIAL,
        is_initialize=is_initialize)
    initial_df.to_csv("biomodels_subnet_initial.csv", index=False)
    # Handle the cases where the search is truncated
    truncated_idx = initial_df["is_truncated"]
    truncated_df = initial_df[truncated_idx]
    reference_networks = _deserializeBiomodels(truncated_df["reference_name"].values)
    target_networks = _deserializeBiomodels(truncated_df["target_name"].values)
    network_pairs = [NetworkPair(r, t) for r, t in zip(reference_networks, target_networks)]
    # Process the truncated networks
    final_df = pd.DataFrame()
    if not is_initialize:
        if os.path.isfile(OUTPUT_CSV):
            final_df = pd.read_csv(OUTPUT_CSV)
    if final_df.empty:
        final_df = initial_df[~truncated_idx].copy()
    existing_reference_networks = final_df["reference_name"].values
    existing_target_networks = final_df["target_name"].values
    for network_pair in network_pairs:
        sel = [(network_pair[0].network_name == r) and (network_pair[1].network_name == t) for r, t in zip(existing_reference_networks, existing_target_networks)]
        is_skip = any(sel)
        if is_skip:
            continue
        is_skip = False
        for network in network_pair:
            if network.network_name in SKIP_NETWORKS:
                is_skip = True
                continue
        if is_skip:
            continue
        new_df = SubnetFinder([network_pair], identity=cn.ID_WEAK).find(
              is_report=True, max_num_assignment=MAX_NUM_ASSIGNMENT_FINAL)
        final_df = pd.concat([final_df, new_df])
        final_df.to_csv(OUTPUT_CSV, index=False)
    # Create the final DataFrame
    final_df.to_csv(OUTPUT_CSV, index=False)

def _deserializeBiomodels(names:List[str])->List[Network]:
    """Deserializes the named BioModels networks

    Args:
        names: str

    Returns:
        List[Network]
    """
    serializer = ModelSerializer(serialization_path=cn.BIOMODELS_SERIALIZATION_PATH)
    networks = serializer.deserialize(model_names=names).networks
    return networks


if __name__ == "__main__":
    freeze_support()
    main(is_initialize=False, identity=cn.ID_STRONG)