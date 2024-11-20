'''Wraps parallel processing around SubnetFinder.'''

'''
Parallelism is achived by running many tasks in parallel. 

manageTasks: Setups up the environment and runs the tasks
executeTask: Processes a task

Tasks receive work from a Queue. The special case of 1 task is run in a single process.
'''


import sirn.constants as cn # type: ignore
from sirn.subnet_finder import SubnetFinder # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
from sirn.parallel_subnet_finder_worker import executeTask, WorkerCheckpointManager, Workunit
from sirn.checkpoint_manager import CheckpointManager
from sirn.mock_queue import MockQueue
from sirn.network import Network

import itertools
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional

CHECKPOINT_FILE = "parallel_subset_finder_checkpoint.txt"
SERIALIZATION_FILE = "serialized.txt"
REFERENCE_SERIALIZATION_FILENAME = "reference_serialized.txt"
TARGET_SERIALIZATION_FILENAME = "target_serialized.txt"
CHECKPOINT_PATH = os.path.join(cn.DATA_DIR, CHECKPOINT_FILE)
# BioModels
BIOMODELS_SERIALIZATION_PATH = os.path.join(cn.DATA_DIR, "biomodels_serialized.txt")   # Serialized BioModels
BIOMODELS_SERIALIZATION_TARGET_FILENAME = 'biomodels_serialized_target.txt'  # Serialized target models
BIOMODELS_SERIALIZATION_REFERENCE_FILENAME = 'biomodels_serialized_reference.txt' # Serialized reference models
BIOMODELS_CHECKPOINT_FILENAME = "biomodels_checkpoint.csv"
BIOMODELS_CHECKPOINT_PATH = os.path.join(cn.DATA_DIR, BIOMODELS_CHECKPOINT_FILENAME)


############################### CLASSES ###############################
class ParallelSubnetFinder(object):
    # Finds subnets of target models for reference models

    def __init__(self, reference_serialization_path:str, target_serialization_path:str,
          identity:str=cn.ID_WEAK,
          checkpoint_path:str=CHECKPOINT_PATH)->None:
        """
        Args:
            reference_serialization_path (str): Path to the reference model serialization file
            target_serialization_path (str): Path to the target model serialization file
            identity (str): Identity type
            checkpoint_path (str): Path to the output serialization file that checkpoints results
               for all tasks.
        """
        self.reference_serialization_path = reference_serialization_path
        serializer = ModelSerializer(None, reference_serialization_path)
        self.reference_networks = serializer.deserialize().networks
        self.target_serialization_path = target_serialization_path
        serializer = ModelSerializer(None, target_serialization_path)
        self.target_networks = serializer.deserialize().networks
        self.identity = identity
        self.checkpoint_path = checkpoint_path
    
    @classmethod
    def findFromNetworks(cls, reference_networks:List[Network],
          target_networks:List[Network],
          identity:str=cn.ID_WEAK,
          is_report:bool=True,
          serialization_dir:str=cn.DATA_DIR,
          is_initialize:bool=False,
          num_task:int=1, checkpoint_path:str=CHECKPOINT_PATH)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_networks (List[Network]): List of reference networks
            target_networks (List[Network]): List of target networks
            identity (str): Identity type
            serialization_dir (str): Directory for serialization files for reference and target
            is_report (bool): If True, report progress
            is_initialize (bool): If True, initialize the checkpoint
            num_task (int): Number of tasks processing tasks
            checkpoint_path (str): Path to the output serialization file that checkpoints results

        Returns:
            pd.DataFrame: (See find)
        """
        #####
        def serializeNetworks(networks, filename:str)->str:
            """
            Serializes the networks to a file, returning the serialization path.

            Args:
                networks (List[Network]): List of networks
                filename (str): Filename

            Returns:
                str: Serialization path
            """
            serialization_path = os.path.join(serialization_dir, filename)
            if not os.path.exists(serialization_path):
                serializer = ModelSerializer(serialization_dir, serialization_path)
                serializer.serializeNetworks(networks)
            return serialization_path
        #####
        target_serialization_path = serializeNetworks(target_networks, TARGET_SERIALIZATION_FILENAME)
        reference_serialization_path = serializeNetworks(reference_networks, REFERENCE_SERIALIZATION_FILENAME)
        #
        finder = cls(reference_serialization_path, target_serialization_path,
          identity=identity, num_task=num_task, checkpoint_path=checkpoint_path)
        return finder.parallelFind(is_report=is_report, is_initialize=is_initialize)
    
    @classmethod
    def findFromDirectories(cls, reference_directory, target_directory, identity:str=cn.ID_WEAK,
          checkpoint_path:str=CHECKPOINT_PATH, num_task:int=1, is_initialize:bool=False,
          is_report:bool=True)->pd.DataFrame:
        """ 
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_directory (str): Directory that contains the reference model files (or serialization path)
            target_directory (str): Directory that contains the target model files or a serialization file (".txt")
            identity (str): Identity type
            num_task (int): Number of tasks processing tasks
            checkpoint_path (str): Path to the output serialization file that checkpoints results
            is_initialize (bool): If True, initialize the checkpoint
            is_report (bool): If True, report progress

        Returns:
            pd.DataFrame: (See find)
        """
        #####
        def getSerializationPath(directory:str)->str:
            """ 
            Obtains the networks from a directory or serialization file.

            Args:
                directory (str): directory path or path to serialization file

            Returns:
                serialization path
            """
            if directory.endswith(".txt"):
                serialization_path = directory
            else:
                # Construct the serialization file path and file
                serialization_path = os.path.join(directory, SERIALIZATION_FILE)
            return serialization_path
        #####
        reference_serialization_path = getSerializationPath(reference_directory)
        target_serialization_path = getSerializationPath(target_directory)
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(reference_serialization_path, target_serialization_path,
          identity=identity, num_task=num_task, checkpoint_path=checkpoint_path)
        return finder.parallelFind(is_report=is_report, is_initialize=is_initialize)
    
    def parallelFind(self, total_process:int=1, is_report:bool=True, is_initialize:bool=True,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->pd.DataFrame:
        """
        Finds reference networks that are subnets of the target networks using parallel processing.

        Args:
            is_report (bool): If True, report progress
            is_initialize (bool): If True, initialize the checkpoint
            max_num_assignment (int): Maximum number of assignments to process

        Returns:
            pd.DataFrame: (See find)
        """
        # Recover checkpoint results
        manager = CheckpointManager(self.checkpoint_path, is_initialize=is_initialize, is_report=is_report)
        df = manager.recover()
        reference_dct = {n.network_name: idx for idx, n in enumerate(self.reference_networks)}
        target_dct = {n.network_name: idx for idx, n in enumerate(self.target_networks)}
        if len(df) > 0:
            completed_workunits = [Workunit(reference_dct[ref_name], target_dct[tar_name])
                  for ref_name, tar_name in zip(df[cn.FINDER_REFERENCE_NETWORK], df[cn.FINDER_TARGET_NETWORK])]
        else:
            completed_workunits = []
        pair_iterator = itertools.product(range(len(self.reference_networks)), range(len(self.target_networks)))
        workunits = [Workunit(i, j) for i, j in pair_iterator]
        workunits = [w for w in workunits if w not in completed_workunits]
        # Set up the task work queue
        if total_process < 0:
            total_process = mp.cpu_count()
        if total_process == 1:
            queueClass = MockQueue
        else:
            queueClass = mp.Manager
        with queueClass() as manager:
            if total_process == 1:
                queue = manager
            else:
                queue = manager.Queue()
            # Populate the queue
            for workunit in workunits:
                queue.put(workunit)
            for _ in range(total_process):
                queue.put(Workunit(is_done=True))
            # Start the processes
            if is_report:
                print(f"**Starting {total_process} processes.")
            args = [(task_idx, queue, total_process, self.checkpoint_path,
                self.reference_serialization_path, self.target_serialization_path,
                     self.identity, is_report, is_initialize, max_num_assignment)
               for task_idx in range(total_process)]
            process_args = zip(*args)
            if queueClass == MockQueue:
                executeTask(*args[0])
            else:
                with ProcessPoolExecutor(max_workers=total_process) as executor:
                    result = executor.map(executeTask, *process_args)
            if is_report:
                print(f"**{total_process} processes completed.")
        # Merge the results
        merged_checkpoint_result = WorkerCheckpointManager.merge(
              self.checkpoint_path, total_process, is_report=is_report)
        # Return the results
        df = merged_checkpoint_result.merged_checkpoint_manager.recover().full_df
        if is_report:
            msg = f"**Done. Processed {merged_checkpoint_result.num_merged_network}"
            msg += " reference networks in {outpath}."
            print(msg)
        return df
    
    @classmethod
    def biomodelsFind(cls,
          reference_network_size:int=15,
          max_num_reference_network:int=-1,
          max_num_target_network:int=-1,
          identity:str=cn.ID_STRONG,
          reference_network_names:List[str]=[],
          is_report:bool=True,
          checkpoint_path:Optional[str]=None,
          num_task:int=-1,
          is_no_boundary_network:bool=True,
          skip_networks:Optional[List[str]]=None,
          is_initialize:bool=True)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.
        The DataFrame returned includes a reference network, target network pair with null strings for found networks
        so that there is a record of all comparisons made.

        Args:
            reference_network_size (int): Size of networks in Bionetworks that are used as reference network
            max_num_reference_network (int): Maximum number of reference networks to process. If -1, process all
            max_num_target_network (int): Maximum number of target networks to process. If -1, process all
            reference_network_names (List[str]): List of reference network names to process
            is_report (bool): If True, report progress
            checkpoint_path (Optional[str]): Path for the checkpoint file
            num_task (int): Number of tasks to process the reference networks (-1 is all CPUs)
            skip_networks (List[str]): List of reference network to skip
            is_filter_unitnetworks (bool): If True, remove networks that only have reactions with one species
            is_initialize (bool): If True, initialize the checkpoint

        Returns:
            pd.DataFrame: (See find)
        """
        if checkpoint_path is None:
            checkpoint_path = BIOMODELS_CHECKPOINT_PATH
        # Select the networks to be analyzed
        if max_num_reference_network == -1:
            max_num_reference_network = int(1e9)
        if max_num_target_network == -1:
            max_num_target_network = int(1e9)
        # Construct the reference and target networks
        serializer = ModelSerializer(None, BIOMODELS_SERIALIZATION_PATH)
        collection = serializer.deserialize()
        all_networks = collection.networks
        if skip_networks is not None:
            all_networks = [n for n in all_networks if n.network_name not in skip_networks]
        if is_no_boundary_network:
            all_networks = [n for n in all_networks if not cls.isBoundaryNetwork(n)]
        if len(reference_network_names) > 0:
            reference_networks = [n for n in all_networks if n.network_name in reference_network_names]
            if len(reference_networks) != len(reference_network_names):
                missing_networks = set(reference_network_names) - set([n.network_name for n in reference_networks])
                raise ValueError(f"Could not find reference networks {missing_networks}.")
        else:
            reference_networks = [n for n in all_networks if n.num_reaction <= reference_network_size]
            num_reference_network = min(len(reference_networks), max_num_reference_network)
            reference_networks = reference_networks[:num_reference_network]
        target_networks = [n for n in all_networks if n.num_reaction > reference_network_size]
        num_target_network = min(len(target_networks), max_num_target_network)
        target_networks = target_networks[:num_target_network]
        # Process the selected networks
        return cls.findFromNetworks(
          reference_networks,
          target_networks,
          identity=identity,
          is_report=is_report,
          is_initialize=is_initialize,
          num_task=num_task, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    df = SubnetFinder.findBiomodelsSubnet(reference_network_size=10, is_report=True)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnet.csv"))
    print("Done")