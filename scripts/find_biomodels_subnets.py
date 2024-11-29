"""Analyzes biomodels for subnets."""

import sirn.constants as cn # type: ignore
from sirn.parallel_subnet_finder import ParallelSubnetFinder  # type: ignore
from sirn.subnet_finder import SubnetFinder  # type: ignore

import numpy as np
from multiprocessing import freeze_support


MAX_NUM_ASSIGNMENT_INITIAL = 1000
MAX_NUM_ASSIGNMENT_FINAL = np.int64(1e12)

def main(is_initialize:bool=False)->None:
    # Find subnets of the Biomodels database
    initial_df = ParallelSubnetFinder.findBiomodelsSubnet(
        reference_model_size=10,
        identity=cn.ID_STRONG,
        is_report=True,
        serialization_dir=cn.DATA_DIR,
        checkpoint_path=None,
        is_no_boundary_network=True,
        num_worker=-1,
        skip_networks=["BIOMD0000000192"],
        batch_size=1,
        max_num_assignment=MAX_NUM_ASSIGNMENT_INITIAL,
        is_initialize=is_initialize)
    initial_df.to_csv("biomodels_subnet_initial.csv", index=False)
    # Handle the cases where the search is truncated
    truncated_idx = initial_df["is_truncated"]
    truncated_df = initial_df[truncated_idx]
    network_pairs = [SubnetFinder.NetworkPair(r, t) for r, t
          in zip(truncated_df["reference_network"], truncated_df["induced_network"])]
    incremental_df = SubnetFinder(network_pairs, identity=cn.ID_STRONG).find(
        is_report=True, max_num_assignment=MAX_NUM_ASSIGNMENT_FINAL)
    # Create the final DataFrame
    final_df = initial_df.copy()
    final_df[truncated_idx] = incremental_df
    final_df.to_csv("biomodels_subnet_final.csv", index=False)

if __name__ == "__main__":
    freeze_support()
    main(is_initialize=False)