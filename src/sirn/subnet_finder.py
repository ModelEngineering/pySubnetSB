'''Finds subnets the pairs of reference networks that are subnets of target networks.'''

import sirn.constants as cn # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn
from sirn.assignment_pair import AssignmentPair # type: ignore

import json
import os
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Tuple, Optional


############################### CLASSES ###############################
class SubnetFinder(object):

    def __init__(self, reference_networks:List[Network], target_networks:List[Network],
                 identity:str=cn.ID_WEAK, num_process:int=-1)->None:
        """
        Args:
            reference_model_directory (str): Directory that contains the reference model files
            target_model_directory (str): Directory that contains the target model files
            identity (str): Identity type
            num_process (int): Number of processes to use. If -1, use all available processors.
        """
        self.reference_networks = reference_networks
        self.target_networks = target_networks
        self.identity = identity
        self.num_process = num_process

    def find(self, is_report:bool=True, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            is_report (bool): If True, report progress
            max_num_assignment (int): Maximum number of assignment pairs

        Returns
            pd.DataFrame: Table of matching networks (cn.FINDER_DATAFRAME_COLUMNS)
                reference_name (str): Reference model name
                target_name (str): Target model name
                reference_network (str): string representation of the reference network
                induced_network (str): string representation of the induced network in the target
                name_dct (dict): Dictionary of mapping of target names to reference names for species and reactions
                                 as a JSON string.
                num_assignment_pair (int): Number of assignment pairs returned
                                            for matching target to subnet
                is_trucated (bool): True if the search is truncated
        """
        dct:dict = {k: [] for k in cn.FINDER_DATAFRAME_COLUMNS}
        for reference_network in self.reference_networks:
            if is_report:
                print(f"Processing reference model: {reference_network.network_name}")
            for target_network in self.target_networks:
                result = reference_network.isStructurallyIdentical(target_network, identity=self.identity,
                      num_process=self.num_process, is_report=is_report, is_subset=True,
                      max_num_assignment=max_num_assignment)
                dct[cn.FINDER_REFERENCE_NAME].append(reference_network.network_name)
                dct[cn.FINDER_TARGET_NAME].append(target_network.network_name)
                dct[cn.FINDER_IS_TRUNCATED].append(result.is_truncated)
                if not result:
                    dct[cn.FINDER_REFERENCE_NETWORK].append(cn.NULL_STR)
                    dct[cn.FINDER_INDUCED_NETWORK].append(cn.NULL_STR)
                    dct[cn.FINDER_NUM_ASSIGNMENT_PAIR].append(cn.NULL_STR)
                    dct[cn.FINDER_NAME_DCT].append(cn.NULL_STR)
                else:
                    # Construct the induced subnet
                    species_assignment_arr = result.assignment_pairs[0].species_assignment
                    reaction_assignment_arr = result.assignment_pairs[0].reaction_assignment
                    species_names = target_network.species_names[species_assignment_arr]
                    reaction_names = target_network.reaction_names[reaction_assignment_arr]
                    network_name = f"{reference_network.network_name}_{target_network.network_name}"
                    induced_network = Network(reference_network.reactant_nmat.values, reference_network.product_nmat.values,
                          reaction_names=reaction_names, species_names=species_names,
                          network_name=network_name)
                    if is_report:
                        print(f"Found matching model: {reference_network.network_name} and {target_network.network_name}")
                    dct[cn.FINDER_REFERENCE_NETWORK].append(str(reference_network))
                    dct[cn.FINDER_INDUCED_NETWORK].append(str(induced_network))
                    dct[cn.FINDER_NUM_ASSIGNMENT_PAIR].append(len(result.assignment_pairs))
                    # Create a more complete assignment pair
                    assignment_pair = AssignmentPair(species_assignment=species_assignment_arr,
                            reaction_assignment=reaction_assignment_arr,
                            reference_reaction_names=reference_network.reaction_names,
                            reference_species_names=reference_network.species_names,
                            target_reaction_names=target_network.reaction_names,
                            target_species_names=target_network.species_names)
                    dct_str = json.dumps(assignment_pair.makeNameDct())
                    dct[cn.FINDER_NAME_DCT].append(dct_str)
        df = pd.DataFrame(dct)
        return df
    
    @classmethod
    def findFromDirectories(cls, reference_directory, target_directory, identity:str=cn.ID_WEAK,
          is_report:bool=True)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_directory (str): Directory that contains the reference model files
            target_directory (str): Directory that contains the target model files or a serialization file (".txt")
            identity (str): Identity type
            is_report (bool): If True, report progress

        Returns:
            pd.DataFrame: (See find)
        """
        #####
        def getNetworks(directory:str)->List[Network]:
            """
            Obtains the networks from a directory or serialization file.

            Args:
                directory (str): directory path or path to serialization file

            Returns:
                Networks
            """
            if directory.endswith(".txt"):
                serialization_path = directory
            else:
                # Construct the serialization file path and file
                serialization_path = os.path.join(directory, cn.SERIALIZATION_FILE)
            # Get the networks
                serializer = ModelSerializer(directory, serialization_path)
                if not os.path.exists(serialization_path):
                    serializer.serialize()
                collection = serializer.deserialize()
            return collection.networks
        #####
        reference_networks = getNetworks(reference_directory)
        target_networks = getNetworks(target_directory)
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(reference_networks, target_networks, identity=identity)
        return finder.find(is_report=is_report)


if __name__ == "__main__":
    df = SubnetFinder.findBiomodelsSubnet(reference_network_size=10, is_report=True)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnet.csv"))
    print("Done")