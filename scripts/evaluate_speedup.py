
"""
Evaluation is done on a set of randomly chosen networks with embedded subnets.sum

The experiments are implemented below. They were done with 10 iterations, reference size 5, target size 100.
Results for 1, 2, 4, 8 processes on 10 core Mac M1 (max_num_assignment  10^11) on Macbook:
    77, 59, 38, 34
    1363, 1133, 889, 794
    295, 157, 84, 48
    383, 187, 101, 57
On server 1, 2, 4, 8, 16:
    7951, 3824, 2021, 1149, 658
    564, 291, 149, 85, 52
    82, 49, 25, 17, 13
    77, 50, 34, 28, 28
    818, 416, 227, 121, 73
    240, 138, 79, 50, 41
"""

import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.network import Network # type: ignore

import numpy as np
import pandas as pd # type: ignore
import time
from typing import List, Tuple


IGNORE_TEST = False
IS_PLOT = False
REFERENCE_SIZE = 5 
TARGET_SIZE = 50 

def run(network_pairs:List[Tuple[Network, Network]], num_process: int=1, num_compare: int=1):
    """_Runs subnet discovery

    Args:
        network_pairs ()
        num_process (int, optional): _description_. Defaults to 1.
        num_compare (int, optional): Number of network comparisons in a vectorization
    """
    num_null = 0
    batch_size = num_compare*2*REFERENCE_SIZE**2
    for reference_network, target_network in network_pairs:
        result = reference_network.isStructurallyIdentical(target_network, is_subnet=True,
            is_report=True,
            num_process=num_process,
            max_batch_size=batch_size,
            max_num_assignment=int(1e11), is_all_valid_assignment=False)
        if len(result.assignment_pairs) == 0:
            num_null += 1
        if not (result or result.is_truncated):
            print("Error in the test. Subnet is absent.")
    print(f"num_process: {num_process}, num_null: {num_null}")
        
def main(num_iteration:int=10, num_processors:list[int]=[1, 2, 4, 8], num_compares:list[int]=[1, 10, 100, 1000]):
    # Do timinings
    # Make the networks
    network_pairs:List[Tuple[Network, Network]] = []
    for _ in range(num_iteration):
        result = Network.makeRandomReferenceAndTarget(num_reference_species=REFERENCE_SIZE,
            num_target_species=TARGET_SIZE)
        network_pairs.append([result.reference_network, result.target_network])  # type: ignore
    print("Completed network construction.")
    for num_process in num_processors:
        for num_compare in num_compares:
            start_time = time.time()
            run(network_pairs, num_process=num_process, num_compare=num_compare)
            elapsed_time = time.time() - start_time
            print(f"num_process: {num_process}, num_compare: {num_compare}, elpased_time: {elapsed_time}")

if __name__ == '__main__':
    main(num_iteration=10, num_processors=[1], num_compares=[1, 10, 100, 1000, 10000])