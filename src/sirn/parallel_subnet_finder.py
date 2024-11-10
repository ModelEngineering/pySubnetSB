'''Wraps parallel processing around SubnetFinder.'''

'''
Parallelism is achived by running many tasks in parallel. 

manageTasks: Setups up the environment and runs the tasks
executeTask: Processes a task

Tasks receive work from a Queue. The special case of 1 task is run in a single process.
'''


import sirn.constants as cn # type: ignore
from sirn.subnet_finder import SubnetFinder, NAME_DCT, REFERENCE_NETWORK # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
from sirn.parallel_subnet_finder_worker import executeTask
from sirn.mock_queue import MockQueue

import collections
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Tuple, Optional

QUEUE_DONE = -1
SERIALIZATION_FILE = "collection_serialization.txt"
BIOMODELS_SERIALIZATION_FILENAME = 'biomodels_serialized.txt'
BIOMODELS_SERIALIZATION_TARGET_FILENAME = 'biomodels_serialized_target.txt'  # Serialized target models
BIOMODELS_SERIALIZATION_REFERENCE_FILENAME = 'biomodels_serialized_reference.txt' # Serialized reference models
BIOMODELS_OUTPATH_FILENAME = "biomodels_subnet.csv"


############################### CLASSES ###############################
class ParallelSubnetFinder(object):
    # Finds subnets of target models for reference models

    def __init__(self, reference_networks:List[Network], target_networks:List[Network], identity:str=cn.ID_WEAK,
          num_task:int=1, data_dir:str=cn.DATA_DIR)->None:
        """
        Args:
            reference_networks: List[Network]: List of reference networks
            target_networks: List[Network]: List of target networks
            identity (str): Identity type
            num_task (int): Number of tasks processing tasks
        """
        self.num_task = num_task
        self.data_dir = data_dir
        self.reference_networks = reference_networks
        self.target_networks = target_networks
        self.identity = identity
        self.total_process = int(mp.cpu_count() / self.num_task)  # Number of processes per task
        # Ensure that class variables are initialized
        self.biomodels_serialization_path = os.path.join(data_dir, BIOMODELS_SERIALIZATION_FILENAME)
        self.biomodels_serialization_target_path = os.path.join(data_dir,
              BIOMODELS_SERIALIZATION_TARGET_FILENAME)
        self.biomodels_serialization_reference_path = os.path.join(data_dir,
              BIOMODELS_SERIALIZATION_REFERENCE_FILENAME)
    
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
                serialization_path = os.path.join(directory, SERIALIZATION_FILE)
            # Get the networks
                serializer = ModelSerializer(directory, serialization_path)
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


    @classmethod
    def _makeReferenceTargetSerializations(cls, reference_networks:List[Network], target_networks:List[Network],
          num_task:int)->None:
        """
        Makes serialization files for the tasks. Each task is provided with a subset of the reference models.

        Args:
            reference_networks (List[Network]): List of reference networks
            target_networks (List[Network]): List of target networks
            out_path (Optional[str]): If not None, write the output to this path for CSV file
            num_task (int): Number of tasks to process the reference networks (-1 is all CPUs)
        """
        # Serialize the target models
        serializer = ModelSerializer(cls.DATA_DIR, cls.BIOMODELS_SERIALIZATION_TARGET_PATH)
        serializer.serializeNetworks(target_networks)
        serializer = ModelSerializer(cls.DATA_DIR, cls.BIOMODELS_SERIALIZATION_REFERENCE_PATH)
        serializer.serializeNetworks(reference_networks)
    
    # FIXME: (a) Use generic worker (b) Don't need to construct separate serialization files 
    @classmethod
    def findBiomodelsSubnet(cls,
          reference_network_size:int=15,
          max_num_reference_network:int=-1,
          max_num_target_network:int=-1,
          identity:str=cn.ID_STRONG,
          reference_network_names:List[str]=[],
          is_report:bool=True,
          outpath:Optional[str]=None,
          batch_size:int=10,
          max_num_task:int=-1,
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
            outpath (Optional[str]): Path for the checkpoint file
            batch_size (int): Number of reference networks to process in a batch
            max_num_task (int): Number of tasks to process the reference networks (-1 is all CPUs)
            skip_networks (List[str]): List of reference network to skip
            is_filter_unitnetworks (bool): If True, remove networks that only have reactions with one species
            is_initialize (bool): If True, initialize the checkpoint

        Returns:
            pd.DataFrame: (See find)
        """
        if outpath is None:
            outpath = cls.BIOMODELS_OUTPATH
        # Select the networks to be analyzed
        if max_num_reference_network == -1:
            max_num_reference_network = int(1e9)
        if max_num_target_network == -1:
            max_num_target_network = int(1e9)
        # Construct the reference and target networks
        serializer = ModelSerializer(None, cls.BIOMODELS_SERIALIZATION_PATH)
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
        # Serialize the networks for the tasks
        num_task = mp.cpu_count() if max_num_task == -1 else max_num_task
        reference_networks = np.random.permutation(reference_networks)
        # FIXME: Create paths and serialize reference and target networks
        # FIXME
        # Set up the task work queue
        if num_task == 1:
            queueClass = MockQueue
        else:
            queueClass = mp.Manager().Queue
        with queueClass() as queue:
            # Populate the queue
            for i in range(len(reference_networks)):
                queue.put(i)
            for _ in range(num_task):
                queue.put(cn.QUEUE_DONE)
            # Start the processes
            if is_report:
                print(f"**Starting {num_task} tasks.")
            args = [(task_idx, queue, num_task, identity, is_report, batch_size, outpath, is_initialize)
               for task_idx in range(num_task)]
            with ProcessPoolExecutor(max_workers=num_task) as executor:
                process_args = zip(*args)
                executor.map(executeTask, process_args)
            queue.join()
            if is_report:
                print(f"**{num_task} tasks completed.")
        # Merge the results
        merged_checkpoint_result = cls._mergeCheckpoints(outpath, num_task, is_report=is_report)
        # Return the results
        df = merged_checkpoint_result.manager.recover()
        if is_report:
            msg = f"**Done. Processed {merged_checkpoint_result.num_merged_network}"
            msg += " reference networks in {outpath}."
            print(msg)
        return df


if __name__ == "__main__":
    df = SubnetFinder.findBiomodelsSubnet(reference_network_size=10, is_report=True)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnet.csv"))
    print("Done")