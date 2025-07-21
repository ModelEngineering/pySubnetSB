[![Build](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml/badge.svg)](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml)

# SUBNET DISCOVERY FOR SBML MODELS

# Motivation
Many advances in biomedical research are driven by structural analysis, a study of the interconnections
between elements in biological systems (e.g., identifying drug target and phylogenetic analyses). Structural analysis
appeals because structural information is much easier to obtain than dynamical data such as species concentrations
and reaction fluxes. Our focus is on subnet discovery in chemical reaction networks (CRNs); that is, discovering a
subset of a target CRN that is structurally identical to a reference CRN. Applications of subnet discovery include the
discovery of conserved chemical pathways and the elucidation of the structure of complex CRNs. Although there are
theoretical results for finding subgraphs, we are unaware of tools for CRN subnet discovery. This is in part due to the
special characteristics of CRN graphs, that they are directed, bipartite, hypergraphs.

# Results
We introduces pySubnetSB, an open source python package for discovering subnets represented in the systems
biology markup language (SBML) community standard. pySubnetSB uses a constraint-based approach to discover
subgraphs using techniques that work well for CRNs, and provides considerable speed-up through vectorization and
process-based parallelism. We provide a methodology for evaluating the statistical significance of subnet discovery and
apply pySubnetSB to discovering subnets in more than 100,000 model pairs in the BioModels repository of curated
models.

# Availability
pySubnetSB is installed using

    pip install pySubnetSB

The package has been tested on linux (Ubuntu 22.04), Windows (Windows 10), and Mac OS (14.7.6). For each, tests were run for python 3.9, 3.10, 3.11, and 3.12.

https://github.com/ModelEngineering/pySubnetSB/blob/main/examples/api_basics.ipynb is a Jupyter notebook that demonstrates pySubsetSB capabilities. https://github.com/ModelEngineering/pySubnetSB/blob/main/examples/api_basics_programmatic.py is a translation of this notebook into a Python program that can be downloaded and executed using

    python api_basics_programmatic.py

You should see output similar to the following:

 
    ---------------------SIMPLE EXAMPLE-----------------------------------

    Reference model: 
        R1: S1 -> S2; k1*S1
        R2: S2 -> S3; k2*S2
        
        S1 = 5
        S2 = 0
        k1 = 1
        k2 = 1
        

    Target model: 
        T1: A -> B; k1*A
        T2: B -> C; k2*B
        T3: B + C -> ; k3*B*C
        
        A = 5
        B = 0
        k1 = 1
        k2 = 1
        k3 = 0.2
        

    Searching for a subnet in the target model that is structurally identical to the reference model.
    mapping pairs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1041/1041 [00:00<00:00, 3757547.73it/s]

    Mapping pairs: [species: [0 1 2], reaction: [0 1]]

    The inferred network in the target is: 
    2060080: 3 species, 2 reactions
    T1: A -> B
    T2: B -> C


    ---------------------LONGER EXAMPLE-----------------------------------

    This example will use all cores on your machine. Change 'num_process' if you want to use fewer cores.

    Reference model: 
        J1: $S3 -> S2;  S3*19.3591127845924;
        J2: S0 -> S4 + S0;  S0*10.3068257839885;
        J3: S4 + S2 -> S4;  S4*S2*13.8915863630362;
        J4: S2 -> S0 + S2;  S2*0.113616698747501;
        J5: S4 + S0 -> S4;  S4*S0*0.240788980014622;
        J6: S2 -> S2 + S2;  S2*1.36258363821544;
        J7: S2 + S4 -> S2;  S2*S4*1.37438814584166;
        
        S0 = 2; S1 = 5; S2 = 7; S3 = 10; S4 = 1;
        

    The target model is BioModels 695:  https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1701090001/3/BIOMD0000000695_url.xml

    Looking for inferred networks in the target. This takes about 10 minutes on 2 cores.
    mapping pairs: 100%|███████████████████████████████████████████████████████████████████████████████████| 483649090/483649090 [00:29<00:00, 16350750.64it/s]

    The number of mapping pairs is: 330

    The first inferred network is: 9318913: 4 species, 7 reactions
    R_31: xFinal_2 -> xFinal_1
    R_10:  -> xFinal_8
    R_33: xFinal_1 + xFinal_8 + xFinal_2 -> xFinal_8 + xFinal_2
    R_24: xFinal_2 -> xFinal_3 + xFinal_2
    R_25: xFinal_3 -> 
    R_32: xFinal_1 + xFinal_8 + xFinal_2 -> 2.0 xFinal_1 + xFinal_8 + xFinal_2
    R_12: xFinal_8 -> 


    ---------------------DIRECTORY OF MODELS-----------------------------------


    Analysis of a directory of reference models and a directory of target models.

    Reference models are serialized in: http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/reference_serialized.txt

    Target models are serialized in: http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt
    Processing reference model: BIOMD0000000031
    mapping pairs: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 260/260 [00:00<00:00, 695484.08it/s]
    Found matching model: BIOMD0000000031 and BIOMD0000000170
    Processing reference model: BIOMD0000000031
    mapping pairs: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1040/1040 [00:00<00:00, 808091.17it/s]
    Found matching model: BIOMD0000000031 and BIOMD0000000228
    Processing reference model: BIOMD0000000031
    mapping pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 260/260 [00:00<00:00, 2065376.97it/s]
    Found matching model: BIOMD0000000031 and BIOMD0000000354
    Processing reference model: BIOMD0000000031
    Processing reference model: BIOMD0000000027
    mapping pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 312/312 [00:00<00:00, 1915992.46it/s]
    Found matching model: BIOMD0000000027 and BIOMD0000000170
    Processing reference model: BIOMD0000000027
    mapping pairs: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 312/312 [00:00<00:00, 473452.55it/s]
    Found matching model: BIOMD0000000027 and BIOMD0000000228
    Processing reference model: BIOMD0000000027
    mapping pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 312/312 [00:00<00:00, 3001428.55it/s]
    Found matching model: BIOMD0000000027 and BIOMD0000000354
    Processing reference model: BIOMD0000000027
    Processing reference model: BIOMD0000000121
    Processing reference model: BIOMD0000000121
    Processing reference model: BIOMD0000000121
    Processing reference model: BIOMD0000000121
    mapping pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 416/416 [00:00<00:00, 1486226.97it/s]
    Found matching model: BIOMD0000000121 and BIOMD0000000960

    Summary of results: 
        reference_name      target_name num_mapping_pair
    0   BIOMD0000000031  BIOMD0000000170               48
    1   BIOMD0000000031  BIOMD0000000228              240
    2   BIOMD0000000031  BIOMD0000000354               12
    3   BIOMD0000000031  BIOMD0000000960                 
    4   BIOMD0000000027  BIOMD0000000170               24
    5   BIOMD0000000027  BIOMD0000000228               60
    6   BIOMD0000000027  BIOMD0000000354               12
    7   BIOMD0000000027  BIOMD0000000960                 
    8   BIOMD0000000121  BIOMD0000000170                 
    9   BIOMD0000000121  BIOMD0000000228                 
    10  BIOMD0000000121  BIOMD0000000354                 
    11  BIOMD0000000121  BIOMD0000000960                6


    The example completed successfully.



# Version History
* 1.0.8 7/21/2025  Improved example for using pySubnetSB and revised github actions workflows.
* 1.0.7 7/20/2025  Finalized code and documentation
* 1.0.6 7/19/2025  Workflows for Ubuntu, Windows, Macos and python 3.9,
                   3.10, 3.11, 3.12
* 1.0.5 7/19/2025  Fix install issues with missing modules
* 1.0.2 4/10/2025. ModelSpecification API accepts many kinds of model inputs, Antimony, SBML, roadrunner.
* 1.0.1 4/09/2025. Improved generation of networks with subnets. Use "mapping_pair" in API. Bug fixes.
* 1.0.0 2/27/2025. First beta release.