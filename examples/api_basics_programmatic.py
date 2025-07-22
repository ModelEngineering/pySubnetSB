# PROGRAMMATIC API BASICS

# REQUIRES pySubnetSB

# 
# Many advances in biomedical research are driven by structural analysis, a study of the interconnections
# between elements in biological systems (e.g., identifying drug target based on the structure of chemical pathways).
# Structural analysis appeals because such information is generally much easier to obtain than dynamical data such as
# species concentrations and reaction fluxes. Our focus is on subnet discovery in chemical reaction networks (CRNs);
# that is, discovering a subset of a target CRN that is structurally identical to a reference CRN. Applications of subnet
# discovery include the discovery of conserved chemical pathways and the elucidation of the structure of complex CRNs.
# Although there are theoretical results for finding subgraphs, we are unaware of tools for CRN subnet discovery. This is
# in part due to the special characteristics of CRN graphs, that they are directed, bipartite, hypergraphs.
# 
# pySubnetSB is an open source Python package for discovering subnets represented in the systems
# biology markup language (SBML) community standard. pySubnetSB uses a constraint-based approach to discover
# subgraphs using techniques that work well for CRNs, and provides considerable speed-up through vectorization and
# process-based parallelism. We provide a methodology for evaluating the statistical significance of subnet discovery, and
# apply pySubnetSB to discovering subnets in more than 200,000 model pairs in the BioModels curated repository


import pySubnetSB # type: ignore
from pySubnetSB.api import ModelSpecification, findReferenceInTarget, findReferencesInTargets, makeSerializationFile  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import multiprocessing as mp  # Do not remove.
import tellurium as te # type: ignore


def main(is_plot:bool=True):
    
    ########################## SIMPLE EXAMPLE ########################## 
    print("\n\n---------------------SIMPLE EXAMPLE-----------------------------------")
    # A one-shot usage is doing discovery of one reference model in one target model.
    # Below, we do a one-shot analysis for two simple Antimony models.
    
    # Reference model is a simple CRN with two reactions and three species.
    reference_model = """
    R1: S2 -> S3; k2*S2
    R2: S1 -> S2; k1*S1
    
    S1 = 5
    S2 = 0
    k1 = 1
    k2 = 1
    """
    print(f"\nReference model: {reference_model}")

    # Target model is contains an inferred network that is structurally identical to the reference model.
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
    print(f"\nTarget model: {target_model}")
    # In the next cell, we do a one-shot search for a subnet in the target model
    #   that is structurally identical to the reference model.
    print("\nSearching for a subnet in the target model that is structurally identical to the reference model.")
    result = findReferenceInTarget(reference_model, target_model)
    print (f"\nMapping pairs: {result.mapping_pairs}")
    print(f"\nThe inferred network in the target is: \n{result.makeInducedNetwork()}")

    ########################## LONGER EXAMPLE ########################## 
    print("\n\n---------------------LONGER EXAMPLE-----------------------------------")
    print("\nThis example will use all cores on your machine. Change 'num_process' if you want to use fewer cores.")
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
    print(f"\nReference model: {reference_model}")
    URL = "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1701090001/3/BIOMD0000000695_url.xml"
    print("\nThe target model is BioModels 695: ", URL)
    print("\nLooking for inferred networks in the target. This takes about 10 minutes on 2 cores.")
    result = findReferenceInTarget(
            reference_model,                                        # Reference model as an Antimony string
            ModelSpecification(URL, specification_type="sbmlurl"),  # Target as a URL in BioModels
            max_num_mapping_pair=1e14,                              # Maximum number of mapping pairs considered
            is_subnet=True,                                         # Checking for subnet, not equality
            num_process=mp.cpu_count(),                             # Use all cores on the machine
            identity=cn.ID_WEAK,                                    # Look for weak identity
            is_report=is_plot)                                      # Provide status information
    
    

    print(f"\nThe number of mapping pairs is: {len(result.mapping_pairs)}")
    print(f"\nThe first inferred network is: {result.makeInducedNetwork(1)}")

    ##################### ANALYZING A DIRECTORY OF MODELS #####################
    # ``pySubnetSB`` can search for subsets in directories of models. Given a a directory of reference models
    # and a directory of target models, each target model is searched for each reference model. A "directory" can be a folder in a file system or it can be a file containing a serialization of such a directory (one serialized network per line, as described in the next section). The file can be in the local file system or a URL to a file on the Internet. In the example below, the serializations of references and targets are URLs in the ``github`` repository for ``pySubnetSB``.
    # 
    # A serialization of biomodels is [here](http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/data/biomodels_serialized.txt).
    print("\n\n---------------------DIRECTORY OF MODELS-----------------------------------")
    reference_url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/reference_serialized.txt"
    target_url = "http://raw.githubusercontent.com/ModelEngineering/pySubnetSB/main/examples/target_serialized.txt"
    print(f"\n\nAnalysis of a directory of reference models and a directory of target models.")
    print(f"\nReference models are serialized in: {reference_url}")
    print(f"\nTarget models are serialized in: {target_url}")
    result_df = findReferencesInTargets(reference_url, target_url, is_report=is_plot)
    columns = ["reference_name", "target_name", "num_mapping_pair"]
    print(f"\nSummary of results: \n{result_df[columns]}")

    ################## SUCCESSFUL COMPLETION OF THE EXAMPLE ##################
    print("\n\nThe example completed successfully.")


if __name__ == '__main__':
    main()
    mp.freeze_support()
