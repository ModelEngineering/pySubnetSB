'''Analyzes Model 10 in BioModels, Kholodenko et al. 2002'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network # type: ignore

import pandas as pd # type: ignore
import numpy as np
from typing import List, Optional, Union
from multiprocessing import freeze_support

# The first URL in the Kholodenko paper
KHOLODENKO = 10
url_dct = {10: "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL6615119181/4/BIOMD0000000010_url.xml",
           146: "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL8256371999/3/BIOMD0000000146_url.xml",
           270: "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1001120000/4/BIOMD0000000270_url.xml",
           466: "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1302180005/3/BIOMD0000000466_url.xml",
           468: "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1308190000/4/BIOMD0000000468_url.xml"
          }
reference_net = Network.makeFromSBMLFile(url_dct[KHOLODENKO])
target_dct = {k: Network.makeFromSBMLFile(v) for k, v in url_dct.items() if k != KHOLODENKO}
#
def main():
    result_dct:dict = {}
    #for model_num, network in target_dct.items():
    for model_num in [146, 270, 466, 468]:
        print(f"***{model_num}***")
        target_net = target_dct[model_num]
        result_dct[model_num] = reference_net.isStructurallyIdentical(target_net, is_subnet=True,
            identity=cn.ID_STRONG, max_num_assignment=1e18)


if __name__ == "__main__":
    freeze_support()
    main()
    import pdb; pdb.set_trace()