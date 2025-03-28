
"""
Evaluation is done on a set of randomly chosen networks with embedded subnets.sum
"""

import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.network import Network # type: ignore

import numpy as np
import pandas as pd # type: ignore
import time


IGNORE_TEST = False
IS_PLOT = False
REFERENCE_SIZE = 10 
TARGET_SIZE = 50 

def run(network_pairs, num_process: int=1):
    num_null = 0
    for reference_network, target_network in network_pairs:
        result = reference_network.isStructurallyIdentical(target_network, is_subnet=True,
            is_report=True, num_process=num_process,
            max_num_assignment=int(1e9), is_all_valid_assignment=False)
        if len(result.assignment_pairs) == 0:
            import pdb; pdb.set_trace()
            num_null += 1
        if not (result or result.is_truncated):
            import pdb; pdb.set_trace()
            raise ValueError("Error in the test")
    print(f"num_process: {num_process}, num_null: {num_null}")
        
def main(num_iteration:int=10):
    # Do timinings
    # Make the networks
    network_pairs = []
    for _ in range(num_iteration):
        result = Network.makeRandomReferenceAndTarget(num_reference_species=REFERENCE_SIZE,
            num_target_species=TARGET_SIZE)
        network_pairs.append([result.reference_network, result.target_network])
    print("Completed network construction.")
    for num_process in [1, 2, 4, 8]:
        start_time = time.time()
        run(network_pairs, num_process=num_process)
        elapsed_time = time.time() - start_time
        print(f"num_process: {num_process}, elpased_time: {elapsed_time}")

if __name__ == '__main__':
    main(num_iteration=10)