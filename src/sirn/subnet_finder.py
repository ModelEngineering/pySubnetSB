'''Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.'''

import sirn.constants as cn # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn
from sirn.checkpoint_manager import CheckpointManager # type: ignore
from sirn.assignment_pair import AssignmentPair # type: ignore

import json
import os 
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import List, Tuple

SERIALIZATION_FILE = "collection_serialization.txt"
BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/SBMLModels/data"
BIOMODELS_SERIALIZATION_PATH = os.path.join(cn.DATA_DIR, 'biomodels_serialized.txt')
BIOMODELS_OUT_PATH = os.path.join(cn.DATA_DIR, "biomodels_subnets.csv")
# Columns
REFERENCE_MODEL = "reference_model"
TARGET_MODEL = "target_model"
REFERENCE_NETWORK = "reference_network"
INDUCED_NETWORK = "induced_network"
NAME_DCT = "name_dct"  # Dictionary of mapping of target names to reference names for species and reactions
NUM_ASSIGNMENT_PAIR = "num_assignment_pair"
COLUMNS = [REFERENCE_MODEL, TARGET_MODEL, REFERENCE_NETWORK, INDUCED_NETWORK, NAME_DCT, NUM_ASSIGNMENT_PAIR]

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
                reference_network (str): string representation of the reference network
                induced_network (str): string representation of the induced network in the target
                name_dct (dict): Dictionary of mapping of target names to reference names for species and reactions
                                 as a JSON string.
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
                dct[REFERENCE_NETWORK].append(cn.NULL_STR)
                dct[INDUCED_NETWORK].append(cn.NULL_STR)
                dct[NAME_DCT].append(cn.NULL_STR)
                dct[NUM_ASSIGNMENT_PAIR].append(cn.NULL_STR)
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
                    dct[NUM_ASSIGNMENT_PAIR].append(len(result.assignment_pairs))
                    # Create a more complete assignment pair
                    assignment_pair = AssignmentPair(species_assignment=species_assignment_arr,
                            reaction_assignment=reaction_assignment_arr,
                            reference_reaction_names=reference.reaction_names,
                            reference_species_names=reference.species_names,
                            target_reaction_names=target.reaction_names,
                            target_species_names=target.species_names)
                    dct_str = json.dumps(assignment_pair.makeNameDct())
                    dct[NAME_DCT].append(dct_str)
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
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str):
                target_network (str):
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
                serialization_path = os.path.join(directory, SERIALIZATION_FILE)
            # Get the networks
                serializer = ModelSerializer(directory, model_parent_dir=None, serialization_path=serialization_path)
                if os.path.exists(serialization_path):
                    collection = serializer.deserialize()
                else:
                    collection = serializer.serialize()
            return collection.networks
        #####
        reference_networks = getNetworks(reference_directory)
        target_networks = getNetworks(target_directory)
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(reference_networks, target_networks, identity=identity)
        return finder.find(is_report=is_report)

    @staticmethod
    def isBoundaryNetwork(network:Network)->bool:
        """
        A boundary network is one where all reactions are either synthesis or degradation of a single species.

        Args:
            network (Network): Network instance

        Returns:
            bool: True if the network has only one species
        """
        reactant_sum_arr = network.reactant_nmat.values.sum(axis=0)
        product_sum_arr = network.product_nmat.values.sum(axis=0)
        is_boundary = np.all((reactant_sum_arr + product_sum_arr) <= 1)
        return bool(is_boundary)

    @classmethod
    def findBiomodelsSubnet(cls,
          reference_model_size:int=15,
          max_num_reference_model:int=-1,
          max_num_target_model:int=-1,
          reference_model_names:List[str]=[],
          is_report:bool=True,
          out_path:str=BIOMODELS_OUT_PATH,
          batch_size:int=10,
          is_no_boundary_network:bool=True,
          is_initialize:bool=True)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.
        The DataFrame returned includes a reference model, target model pair with null strings for found networks
        so that there is a record of all comparisons made.

        Args:
            reference_model_size (int): Size of models in BioModels that are used as reference model
            max_num_reference_model (int): Maximum number of reference models to process. If -1, process all
            max_num_target_model (int): Maximum number of target models to process. If -1, process all
            reference_model_names (List[str]): List of reference model names to process
            is_report (bool): If True, report progress
            out_path (Optional[str]): If not None, write the output to this path for CSV file
            batch_size (int): Number of reference models to process in a batch
            is_filter_unitnetworks (bool): If True, remove networks that only have reactions with one species
            is_initialize (bool): If True, initialize the checkpoint

        Returns:
            pd.DataFrame: Table of matching models
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str): may be the null string
                target_network (str): may be the null string 
        """
        if max_num_reference_model == -1:
            max_num_reference_model = int(1e9)
        if max_num_target_model == -1:
            max_num_target_model = int(1e9)
        #
        manager = _CheckpointManager(out_path, is_report=is_report, is_initialize=is_initialize)
        serializer = ModelSerializer(BIOMODELS_DIR, model_parent_dir=None, serialization_path=BIOMODELS_SERIALIZATION_PATH)
        collection = serializer.deserialize()
        all_networks = collection.networks
        if is_no_boundary_network:
            all_networks = [n for n in all_networks if not cls.isBoundaryNetwork(n)]
        if len(reference_model_names) > 0:
            reference_models = [n for n in all_networks if n.network_name in reference_model_names]
            if len(reference_models) != len(reference_model_names):
                missing_models = set(reference_model_names) - set([n.network_name for n in reference_models])
                raise ValueError(f"Could not find reference models {missing_models}.")
        else:
            reference_models = [n for n in all_networks if n.num_reaction <= reference_model_size]
            num_reference_model = min(len(reference_models), max_num_reference_model)
            reference_models = reference_models[:num_reference_model]
        target_models = [n for n in all_networks if n.num_reaction > reference_model_size]
        num_target_model = min(len(target_models), max_num_target_model)
        target_models = target_models[:num_target_model]
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
        num_comparison = len(full_df) - len(stripped_df)
        print(f"**Done. Processed {num_comparison} model comparisons.")
        return stripped_df


##############################################################
class _CheckpointManager(CheckpointManager):
    # Specialization of CheckpointManager to SubnetFinder

    def __init__(self, path:str, is_report:bool=True, is_initialize:bool=False)->None:
        """
        Args:
            subnet_finder (SubnetFinder): SubnetFinder instance
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
        """
        super().__init__(path, is_report=is_report, is_initialize=is_initialize)

    def recover(self)->Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Recovers a previously saved DataFrame. The recovered dataframe deletes entries with model strings that are null.

        Returns:
            pd.DataFrame: DataFrame of the checkpoint
            pd.DataFrame: DataFrame of the checkpoint stripped of null entries
            np.ndarray: List of processed tasks
        """
        df = super().recover()
        if len(df) > 0:
            full_df = pd.read_csv(self.path)
            pruned_df, processed_list = _prune(full_df)
            # Convert the JSON string to a dictionary
            if len(pruned_df) > 0:
                pruned_df[NAME_DCT] = pruned_df[NAME_DCT].apply(lambda x: json.loads(x))
        else:
            full_df = pd.DataFrame()
            pruned_df = pd.DataFrame()
            processed_list = []
        self._print(f"Recovering {len(processed_list)} processed models from {self.path}")
        return full_df, pruned_df, processed_list


if __name__ == "__main__":
    df = SubnetFinder.findBiomodelsSubnet(reference_model_size=10, is_report=True)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnet.csv"))
    print("Done")