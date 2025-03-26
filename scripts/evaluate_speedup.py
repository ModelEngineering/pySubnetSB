
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
REFERENCE_SIZE = 20
TARGET_SIZE =100 

def run(network_pairs, num_process: int=1):
    for reference_network, target_network in network_pairs:
        _ = reference_network.isStructurallyIdentical(target_network, is_subnet=True,
            is_report=False, num_process=num_process)
        
def main(num_iteration:int=10):
    # Do timinings
    # Make the networks
    network_pairs = []
    for _ in range(num_iteration):
        result = Network.makeRandomReferenceAndTarget(num_reference_species=REFERENCE_SIZE,
            num_target_species=TARGET_SIZE)
        network_pairs.append([result.reference_network, result.target_network])
    for num_process in [1, 2, 4, 8]:
        start_time = time.time()
        run(network_pairs, num_process=num_process)
        elapsed_time = time.time() - start_time
        print(f"num_process: {num_process}, elpased_time: {elapsed_time}")

if __name__ == '__main__':
    main()