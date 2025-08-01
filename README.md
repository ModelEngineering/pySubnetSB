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

# pySubnetSB
``pySubnetSB`` is an open source python package for discovering subnets represented in the systems
biology markup language (SBML) community standard. By *subnet discovery* we meaning filnding a specified **reference CRN** in a larger **target CRN**. This is an example of the subgraph finding problem in graph theory, which is very computationally demanding (NP-hard). We exploit special characteristics of CRNs to reduce the computational complexity. We also use a combination of vectorization and process parallelism to achieve considerable speedus.

Below, we summarize the ``pySubnetSB`` API.

## Single Reference, Single Target: Simple Case
Here we illustrate ``pySubnetSB`` for two small networks using default values in the API call.
The reference and target models are:

    rerence_model = """
        R1: S2 -> S3; k2*S2
        R2: S1 -> S2; k1*S1
        
        S1 = 5
        S2 = 0
        k1 = 1
        k2 = 1ef
        """
        
    target_model = """
        T1: A -> B; k1*A
        T2: B -> C; k2*B
        T3: B + C -> ; k3*B*C
        
        A = 5
        B = 0
        k1 = 1
        k2 = 1
        k3 = 0.2
        """

The API call is
    
    result = findReferenceInTarget(reference_model, target_model)

``result.mapping_pairs`` describes how species and reactions in the target are mapped to the reference. This is a list with two elements. 

    Mapping pairs: [species: [1 2 0], reaction: [1 0]]

A mapping pair is a list of two lists. The first list is the species mapping. The i-th position in this list is for the i-th reference species. (Species and reactions are indexed by the sequence in which they are encountered in the model.) The first position in this list (Python index 0) contains a 1. This means that ``S2``, the first reference species (index 0), is mapped to target species ``B`` (target species indexed as 1).  The reaction list indicates that reaction ``R1`` (index 0 in the reference) is mapped to reaction ``T2`` (index 1 in the target).

``result.makeInferredNetwork()`` constructs the inferred network in the target based on the mappings. In this example, the inferred network is

    T2: B -> C
    T1: A -> B

## Single Reference, Single Target: More Advanced
Here, we illustrate ``pySubnetSB`` for a more compute intensive example. The reference model is an oscillating network.

    reference_model = """
        J1: $S3 -> S2;  S3*19.3591127845924;
        J2: S0 -> S4 + S0;  S0*10.3068257839885;
        J3: S4 + S2 -> S4;  S4*S2*13.8915863630362;
        J4: S2 -> S0 + S2;  S2*0.113616698747501;
        J5: S4 + S0 -> S4;  S4*S0*0.240788980014622;
        J6: S2 -> S2 + S2;  S2*1.36258363821544;
        J7: S2 + S4 -> S2;  S2*S4*1.37438814584166;
        
        S0 = 2; S1 = 5; S2 = 7; S3 = 10; S4 = 1;
    """
The target model is BioModels 695.

    URL = "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1701090001/3/BIOMD0000000695_url.xml"
    result = findReferenceInTarget(
            reference_model,
            ModelSpecification(URL, specification_type="sbmlurl"),
            max_num_mapping_pair=1e14,
            num_process=mp.cpu_count(),
            identity=cn.ID_WEAK)

As before, the first two arguments of the API call are the reference and target models. Since the target is a URL, ``ModelSpecification`` is used to convert the URL to a model string. ``max_num_mapping_pair`` is used to manage computational demands by limiting the number mapping pairs that are considered. ``num_process`` specifies the number of processes (cores) that are used by ``pySubnetSB``. By default, all cores are used. Last, ``identity`` specifies the kind of subnet to discover.

The ``identity`` argument requires more explanation. A subnet of the target is weakly identical (``cn.ID_WEAK``) to the reference if they have the same stoichiometry matrix. A target subnet is strongly identical (default) if it is just a renaming of the species and reactions in the reference network.

Running the foregoing code takes about 10 minutes on a two core machine. You will see a status bar as the command executes that indicates the number of mapping pairs processed.

    mapping pairs: 100%|███████████████████████████████████████████████████████████████████████████████████| 483649090/483649090 [00:29<00:00, 16350750.64it/s]

As before ``result.mapping_pairs`` is a list of mapping pairs. You can display the inferred network form mapping pair 1 using ``result.makeInducedNetwork(1)``. There is some stochasticity to the order of the results. When we ran this study, inferred network 1 was:

    R_31: xFinal_2 -> xFinal_1
    R_10:  -> xFinal_8
    R_33: xFinal_1 + xFinal_8 + xFinal_2 -> xFinal_8 + xFinal_2
    R_24: xFinal_2 -> xFinal_3 + xFinal_2
    R_25: xFinal_3 -> 
    R_32: xFinal_1 + xFinal_8 + xFinal_2 -> 2.0 xFinal_1 + xFinal_8 + xFinal_2
    R_12: xFinal_8 -> 

This requires some elaboration. Note that although ``pySubnetSB`` matches ``R_10`` in the target with ``J2`` in the reference, the reactions look quite different. These reactions are:

    R_10:  -> xFinal_8
    J2: S0 -> S4 + S0

These reactions are a weakly identical because the match is based on the *stoichiometry matrices*. Recall that the stoichiometry matrix contains the difference between species in the products and those in the reactant. As such, ``S0`` in the reactants is subtracted from ``S0`` in the product and so ``J2`` is equvalent to ``-> S4``, which does look lik ``R_10``.

## Multiple References and Targets
``pySubnetSB`` supports checking for multiple reference networks in multiple target networks.
This can be done by having a directory of reference models and a directory of target models.
``pySubnetSB`` can serialize the structural characteristics of a model into a one line string. (See the discussion of the API call ``makeSerializationFile`` in the Jupyter notebook referenced in the Availability section.) This capability allows you to specify a serialization file instead of a directory, which is often more convenient.

    reference_url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/reference_serialized.txt"
    target_url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt"
    result_df = findReferencesInTargets(reference_url, target_url)

The output of this API is a dataframe with information about the comparisons. Below is the output produced from this analysis for 3 columns in the dataframe.

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


# Availability
pySubnetSB is installed using

    pip install pySubnetSB

The package has been tested on linux (Ubuntu 22.04), Windows (Windows 10), and Mac OS (14.7.6). For each, tests were run for Python 3.9, 3.10, 3.11, and 3.12.

https://github.com/ModelEngineering/pySubnetSB/blob/main/examples/api_basics.ipynb is a Jupyter notebook that demonstrates pySubsetSB capabilities. https://github.com/ModelEngineering/pySubnetSB/blob/main/examples/api_basics_programmatic.py contains much of the code in the notebook. You can test your install of ``pySubnetSB`` by downloading this script and executing it using

    python api_basics_programmatic.py


# Version History
* 1.0.8 7/21/2025  Improved example for using pySubnetSB and revised github actions workflows.
* 1.0.7 7/20/2025  Finalized code and documentation
* 1.0.6 7/19/2025  Workflows for Ubuntu, Windows, Macos and python 3.9,
                   3.10, 3.11, 3.12
* 1.0.5 7/19/2025  Fix install issues with missing modules
* 1.0.2 4/10/2025. ModelSpecification API accepts many kinds of model inputs, Antimony, SBML, roadrunner.
* 1.0.1 4/09/2025. Improved generation of networks with subnets. Use "mapping_pair" in API. Bug fixes.
* 1.0.0 2/27/2025. First beta release.