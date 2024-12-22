"""Analyzes biomodels for subnets."""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.parallel_subnet_finder import ParallelSubnetFinder  # type: ignore
from pySubnetSB.subnet_finder import SubnetFinder, NetworkPair  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore

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
              ]
OUTPUT_CSV = "biomodels_subnet_final.csv"

def main(is_initialize:bool=False)->None:
    # Find subnets of the Biomodels database
    initial_df = ParallelSubnetFinder.biomodelsFind(
        reference_network_size=10,
        identity=cn.ID_STRONG,
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
    final_df = initial_df[~truncated_idx].copy()
    for network_pair in network_pairs:
        new_df = SubnetFinder([network_pair], identity=cn.ID_STRONG).find(
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
    main(is_initialize=False)