'''Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.'''

import sirn.constants as cn # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn
from sirn.checkpoint_manager import CheckpointManager # type: ignore

import os 

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import List, Tuple

SERIALIZATION_FILE = "collection_serialization.txt"
BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/SBMLModels/data"
BIOMODELS_SERIALIZATION_PATH = os.path.join(cn.DATA_DIR, 'biomodels_serialized.txt')
# Columns
REFERENCE_MODEL = "reference_model"
TARGET_MODEL = "target_model"
REFERENCE_NETWORK = "reference_network"
INDUCED_NETWORK = "induced_network"
COLUMNS = [REFERENCE_MODEL, TARGET_MODEL, REFERENCE_NETWORK, INDUCED_NETWORK]
BIOMODELS_OUT_PATH = os.path.join(cn.DATA_DIR, "biomodels_subnets.csv")

############################### INTERNAL FUNCTIONS ###############################
def _prune(df:pd.DataFrame)->Tuple[pd.DataFrame, list]:
    """
    Prunes a DataFrame to include only rows where the reference network is not the null string.

    Args:
        df (pd.DataFrame): Table of matching models
            reference_model (str): Reference model name
            target_model (str): Target model name
            reference_network (str): may be the null string
            target_network (str): may be the null string

    Returns:
        pd.DataFrame: Pruned DataFrame
        list: List of reference models that were pruned
    """
    is_null = df[REFERENCE_NETWORK].isnull()
    is_null_str = df[REFERENCE_NETWORK] == cn.NULL_STR
    not_sel = is_null | is_null_str
    reference_models = list(set(df[not_sel][REFERENCE_MODEL].values))
    return df[~not_sel], reference_models


############################### CLASSES ###############################
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

        Args:
            is_report (bool): If True, report progress

        Returns:
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str): "" if no match
                target_network (str): "" if no match
        """
        dct:dict = {k: [] for k in COLUMNS}
        for reference in self.reference_models:
            if is_report:
                print(f"Processing reference model: {reference.network_name}")
            for target in self.target_models:
                result = reference.isStructurallyIdentical(target, identity=self.identity,
                      is_report=is_report)
                dct[REFERENCE_MODEL].append(reference.network_name)
                dct[TARGET_MODEL].append(target.network_name)
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
                    dct[REFERENCE_NETWORK].append(str(reference))
                    dct[INDUCED_NETWORK].append(str(induced_network))
                else:
                    dct[REFERENCE_NETWORK].append(cn.NULL_STR)
                    dct[INDUCED_NETWORK].append(cn.NULL_STR)
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
        directory_dct = {REFERENCE_MODEL: reference_directory, TARGET_MODEL: target_directory}
        path_dct = {k: makeSerializationFilePath(d) for k,d in directory_dct.items()}
        collection_dct:dict = {}
        for dir_type, path in path_dct.items():
            serializer = ModelSerializer(directory_dct[dir_type], model_parent_dir=None, serialization_path=path)
            if os.path.exists(path):
                collection_dct[dir_type] = serializer.deserialize()
            else:
                collection_dct[dir_type] = serializer.serialize()
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(collection_dct[REFERENCE_MODEL].networks,
              collection_dct[TARGET_MODEL].networks, identity=identity)
        return finder.find(is_report=is_report)
    
    @classmethod
    def findBiomodelsSubsets(cls, reference_model_size:int=15, is_report:bool=True,
          out_path:str=BIOMODELS_OUT_PATH, batch_size:int=10)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.
        The DataFrame returned includes a reference model, target model pair with null strings for found networks
        so that there is a record of all comparisons made.

        Args:
            reference_model_size (int): Size of models in BioModels that are used as reference model
            is_report (bool): If True, report progress
            out_path (Optional[str]): If not None, write the output to this path for CSV file
            batch_size (int): Number of reference models to process in a batch

        Returns:
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str): may be the null string
                target_network (str): may be the null string 
        """
        manager = _CheckpointManager(out_path, is_report=is_report)
        serializer = ModelSerializer(BIOMODELS_DIR, model_parent_dir=None, serialization_path=BIOMODELS_SERIALIZATION_PATH)
        collection = serializer.deserialize()
        all_networks = collection.networks
        reference_models = [n for n in all_networks if n.num_reaction <= reference_model_size]
        target_models = [n for n in all_networks if n.num_reaction > reference_model_size]
        full_df, _, processed_reference_models = manager.recover()
        unprocessed_reference_models = [n for n in reference_models if n.network_name not in processed_reference_models]
        # Process the reference models in batches
        while len(unprocessed_reference_models) > 0:
            reference_model_batch = unprocessed_reference_models[:batch_size]
            unprocessed_reference_models = unprocessed_reference_models[batch_size:]
            finder = cls(reference_model_batch, target_models, identity=cn.ID_STRONG)
            incremental_df = finder.find(is_report=is_report)
            full_df = pd.concat([full_df, incremental_df], ignore_index=True)
            manager.checkpoint(full_df)
            if is_report:
                print(f"**Processed {len(full_df)} models.")
        full_df, stripped_df, _ = manager.recover()
        print(f"**Done. Processed {len(full_df)} models.")
        return stripped_df


##############################################################
class _CheckpointManager(CheckpointManager):
    # Specialization of CheckpointManager to SubnetFinder

    def __init__(self, path:str, is_report:bool=True)->None:
        """
        Args:
            subnet_finder (SubnetFinder): SubnetFinder instance
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
        """
        super().__init__(path, is_report)

    def recover(self)->Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Recovers a previously saved DataFrame. The recovered dataframe deletes entries with model strings that are null.

        Returns:
            pd.DataFrame: DataFrame of the checkpoint
            pd.DataFrame: DataFrame of the checkpoint stripped of null entries
            np.ndarray: List of processed tasks
        """
        if not os.path.exists(self.path):
            full_df = pd.DataFrame()
            pruned_df = pd.DataFrame()
            processed_list:list = []
        else:
            full_df = pd.read_csv(self.path)
            pruned_df, processed_list = _prune(full_df)
        self._print(f"Recovering a dataframe of length {len(full_df)} from {self.path}")
        return full_df, pruned_df, processed_list


if __name__ == "__main__":
    df = SubnetFinder.findBiomodelsSubsets(reference_model_size=10, is_report=True)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnets.csv"))
    print("Done")