'''Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.'''

import sirn.constants as cn # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn

import os 

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import List, Optional

SERIALIZATION_FILE = "collection_serialization.txt"

class SubnetFinder(object):

    def __init__(self, reference_models:List[Network], target_models:List[Network], identity:str=cn.ID_WEAK)->None:
        """
        Args:
            reference_model_directory (str): Directory that contains the reference model files
            target_model_directory (str): Directory that contains the target model files
            identity (str): Identity type
        """
        self.reference_models = reference_models
        self.target_models = target_models
        self.identity = identity

    def find(self, is_report:bool=True)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Returns:
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str):
                target_network (str):
        """
        REFERENCE_MODEL = "reference_model"
        TARGET_MODEL = "target_model"
        REFERENCE_NETWORK = "reference_network"
        INDUCED_NETWORK = "induced_network"
        COLUMNS = [REFERENCE_MODEL, TARGET_MODEL, REFERENCE_NETWORK, INDUCED_NETWORK]
        dct:dict = {k: [] for k in COLUMNS}
        for reference in self.reference_models:
            if is_report:
                print(f"Processing reference model: {reference.network_name}")
            for target in self.target_models:
                result = reference.isStructurallyIdentical(target, identity=self.identity,
                      is_report=is_report)
                if result:
                    # Construct the induced subnet
                    species_assignment_arr = result.assignment_pairs[0].species_assignment
                    reaction_assignment_arr = result.assignment_pairs[0].reaction_assignment
                    species_names = target.species_names[species_assignment_arr]
                    reaction_names = target.reaction_names[reaction_assignment_arr]
                    network_name = f"{reference.network_name}_{target.network_name}"
                    induced_network = Network(reference.reactant_nmat.values, reference.product_nmat.values,
                          reaction_names=reaction_names, species_names=species_names,
                          network_name=network_name)
                    if is_report:
                        print(f"Found matching model: {reference.network_name} and {target.network_name}")
                    dct[REFERENCE_MODEL].append(reference.network_name)
                    dct[TARGET_MODEL].append(target.network_name)
                    dct[REFERENCE_NETWORK].append(str(reference))
                    dct[INDUCED_NETWORK].append(str(induced_network))
        df = pd.DataFrame(dct)
        return df
    
    @classmethod
    def findFromDirectories(cls, reference_directory, target_directory, identity:str=cn.ID_WEAK,
          is_report:bool=True)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_directory (str): Directory that contains the reference model files
            target_directory (str): Directory that contains the target model files
            identity (str): Identity type
            is_report (bool): If True, report progress

        Returns:
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str):
                target_network (str):
        """
        #####
        def makeSerializationFilePath(directory:str)->str:
            return os.path.join(directory, SERIALIZATION_FILE)
        #####
        REFERENCE = "reference"
        TARGET = "target"
        directory_dct = {REFERENCE: reference_directory, TARGET: target_directory}
        path_dct = {k: makeSerializationFilePath(d) for k,d in directory_dct.items()}
        collection_dct:dict = {}
        for dir_type, path in path_dct.items():
            serializer = ModelSerializer(directory_dct[dir_type], model_parent_dir=None, serialization_path=path)
            if os.path.exists(path):
                collection_dct[dir_type] = serializer.deserialize()
            else:
                collection_dct[dir_type] = serializer.serialize()
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(collection_dct[REFERENCE].networks, collection_dct[TARGET].networks, identity=identity)
        return finder.find(is_report=is_report)