'''Creates workunits for workers, starts Worker processes, combines Worker results in their checkpoint files.'''

"""
Parallelism is achived by running workers in parallel. 
"""


import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.model_serializer import ModelSerializer # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.parallel_subnet_finder_worker import executeWorker, Workunit # type: ignore
from pySubnetSB.checkpoint_manager import CheckpointManager # type: ignore
from pySubnetSB.worker_checkpoint_manager import WorkerCheckpointManager # type: ignore
from pySubnetSB.network import Network # type: ignore
from pySubnetSB.subnet_finder_workunit_manager import SubnetFinderWorkunitManager  # type: ignore
from pySubnetSB import util

import itertools
import os
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional

CHECKPOINT_FILE = "parallel_subset_finder_checkpoint.txt"
WORKUNIT_CSV_FILE = "parallel_subset_finder_workunit.csv"
SERIALIZATION_FILE = "serialized.txt"
REFERENCE_SERIALIZATION_FILENAME = "reference_serialized.txt"
TARGET_SERIALIZATION_FILENAME = "target_serialized.txt"
CHECKPOINT_PATH = os.path.join(cn.AUXILIARY_DATA_DIR, CHECKPOINT_FILE)
WORKUNIT_CSV_PATH = os.path.join(cn.AUXILIARY_DATA_DIR, WORKUNIT_CSV_FILE)
CHECKPOINT_INTERVAL = 100 # Number of workunits procesed between checkpoints
BIOMODELS_SERIALIZATION_TARGET_FILENAME = 'biomodels_serialized_target.txt'  # Serialized target models
BIOMODELS_SERIALIZATION_REFERENCE_FILENAME = 'biomodels_serialized_reference.txt' # Serialized reference models
BIOMODELS_CHECKPOINT_FILENAME = "biomodels_checkpoint.csv"
BIOMODELS_CHECKPOINT_PATH = os.path.join(cn.AUXILIARY_DATA_DIR, BIOMODELS_CHECKPOINT_FILENAME)

# Performance monitoring
LAST_TIME = "last_time"
global_dct = {LAST_TIME: time.time()}
#####
def printPerformanceInformation(msg:str, initialize:bool=False):
    # Print performance information
    #if True and not initialize:
    if False:
        last_time = global_dct["last_time"]
        now_time = time.time()
        duration = now_time - last_time
        new_msg = f"**{msg}: {duration} seconds."
        print(new_msg, flush=True)
    global_dct[LAST_TIME]= time.time()
####


############################### CLASSES ###############################
class ParallelSubnetFinder(object):
    # Finds subnets of target models for reference models

    def __init__(self, reference_serialization_path:str, target_serialization_path:str,
          identity:str=cn.ID_WEAK,
          checkpoint_path:str=CHECKPOINT_PATH,
          workunit_csv_path:str=WORKUNIT_CSV_PATH)->None:
        """
        Args:
            reference_serialization_path (str): Path to the reference model serialization file
            target_serialization_path (str): Path to the target model serialization file
            identity (str): Identity type
            checkpoint_path (str): Path to the output serialization file that checkpoints results
               for all tasks.
            workunit_csv_path (str): Path to the CSV file that contains workunits. Overwritten.
        """
        self.reference_serialization_path = reference_serialization_path
        serializer = ModelSerializer(None, reference_serialization_path)
        self.reference_networks = serializer.deserialize().networks
        self.target_serialization_path = target_serialization_path
        serializer = ModelSerializer(None, target_serialization_path)
        self.target_networks = serializer.deserialize().networks
        self.identity = identity
        self.checkpoint_path = checkpoint_path
        self.workunit_csv_path = workunit_csv_path
    
    @classmethod
    def findFromNetworks(cls, reference_networks:List[Network],
          target_networks:List[Network],
          identity:str=cn.ID_WEAK,
          is_report:bool=True,
          serialization_dir:str=cn.DATA_DIR,
          is_initialize:bool=False,
          num_worker:int=-1,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          reference_serialization_filename:str=REFERENCE_SERIALIZATION_FILENAME,
          target_serialization_filename:str=TARGET_SERIALIZATION_FILENAME,
          checkpoint_path:str=CHECKPOINT_PATH)->pd.DataFrame:
        """
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_networks (List[Network]): List of reference networks
            target_networks (List[Network]): List of target networks
            identity (str): Identity type
            serialization_dir (str): Directory for serialization files for reference and target
            is_report (bool): If True, report progress
            is_initialize (bool): If True, initialize the checkpoint
            num_worker (int): Number of workers/CPUs to run (<0 is all CPUs)
            max_num_assignment (int): Maximum number of assignment pairs
            reference_serialization_filename (str): Filename for the reference serialization file
            target_serialization_filename (str): Filename for the target serialization file
            checkpoint_path (str): Path to the output serialization file that checkpoints results

        Returns:
            pd.DataFrame: (See parallelFind)
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
            serializer = ModelSerializer(serialization_dir, serialization_path)
            serializer.serializeNetworks(networks, is_initialize=True)
            return serialization_path
        #####
        target_serialization_path = serializeNetworks(target_networks, target_serialization_filename)
        reference_serialization_path = serializeNetworks(reference_networks,
              reference_serialization_filename)
        #
        finder = cls(reference_serialization_path, target_serialization_path,
          identity=identity, checkpoint_path=checkpoint_path)
        return finder.parallelFind(is_report=is_report, is_initialize=is_initialize,
              max_num_assignment=max_num_assignment, num_worker=num_worker)
    
    @classmethod
    def directoryFind(cls, reference_directory, target_directory, identity:str=cn.ID_WEAK,
          checkpoint_path:str=CHECKPOINT_PATH, num_worker:int=-1, is_initialize:bool=False,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          is_report:bool=True)->pd.DataFrame:
        """ 
        Finds subnets of SBML/Antmony models in a target directory for SBML/Antimony models in a reference directory.

        Args:
            reference_directory (str): Directory that contains the reference model files (or serialization path)
            target_directory (str): Directory that contains the target model files or a serialization file (".txt")
            identity (str): Identity type
            num_worker (int): Number of workers
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
                if not os.path.exists(serialization_path):
                    serializer = ModelSerializer(directory, serialization_path)
                    serializer.serialize()
            return serialization_path
        #####
        reference_serialization_path = getSerializationPath(reference_directory)
        target_serialization_path = getSerializationPath(target_directory)
        # Put the serialized models in the directory. Check for it on invocation.
        finder = cls(reference_serialization_path, target_serialization_path,
          identity=identity, checkpoint_path=checkpoint_path)
        return finder.parallelFind(is_report=is_report, is_initialize=is_initialize,
              num_worker=num_worker, max_num_assignment=max_num_assignment)
    
    def parallelFind(self, num_worker:int=-1, is_report:bool=True, is_initialize:bool=True,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->pd.DataFrame:
        """
        Finds reference networks that are subnets of the target networks using parallel processing.

        Args:
            is_report (bool): If True, report progress
            is_initialize (bool): If True, initialize the checkpoint
            max_num_assignment (int): Maximum number of assignments to process

        Returns:
            pd.DataFrame: (See cn.FINDER_DATAFRAME_COLUMNS)
        """
        #####
        def _print(msg:str):
            if is_report:
                print(msg)
        #####        
        def getArgs(worker_idx:int, is_terminate:bool=True)->tuple:
            is_initialize_worker = True  # Initialize the worker checkpoint
            return worker_idx, self.workunit_csv_path, num_worker, self.checkpoint_path, \
                          self.reference_serialization_path, self.target_serialization_path, \
                          self.identity, is_report, \
                          is_initialize_worker, \
                          max_num_assignment,  \
                          CHECKPOINT_INTERVAL, is_terminate
        #####
        printPerformanceInformation("parallelFind/time0", initialize=True)
        # Initializations
        if num_worker < 0:
            num_worker = mp.cpu_count()
        # Recover checkpoint results
        manager = CheckpointManager(self.checkpoint_path, is_initialize=is_initialize,
              is_report=is_report)
        df = manager.recover()
        if len(df) > 0:
            known_names = set(df[cn.FINDER_REFERENCE_NAME]).intersection([n.network_name for n in self.reference_networks])
            df = df[df[cn.FINDER_REFERENCE_NAME].isin(known_names)]
        reference_dct = {n.network_name: idx for idx, n in enumerate(self.reference_networks)}
        target_dct = {n.network_name: idx for idx, n in enumerate(self.target_networks)}
        if len(df) > 0:
            ref_idx_arr = np.array([reference_dct[str(n)] for n in df[cn.FINDER_REFERENCE_NAME]])
            tgt_idx_arr = np.array([target_dct[str(n)] for n in df[cn.FINDER_TARGET_NAME]])
            completed_encodes = set(util.encodeIntPair(ref_idx_arr, tgt_idx_arr))
        else:
            completed_encodes = set([])
        all_pairs = list(itertools.product(range(len(self.reference_networks)),
              range(len(self.target_networks))))
        all_ref_idx_arr = np.array([i for i, j in all_pairs])
        all_tgt_idx_arr = np.array([j for i, j in all_pairs])
        all_pair_encodeds = set(util.encodeIntPair(all_ref_idx_arr, all_tgt_idx_arr))
        unfinished_arr = np.array(list(all_pair_encodeds - completed_encodes))
        unfinished_pairs = util.decodeIntPair(unfinished_arr)
        workunits = [Workunit(i, j) for i, j in zip(unfinished_pairs[0], unfinished_pairs[1])]
        printPerformanceInformation(f"parallelFind/time1: {len(workunits)} workunits.")
        if len(workunits) > 0:
            # Create the workunits
            workunit_manager = SubnetFinderWorkunitManager(self.workunit_csv_path, num_worker=num_worker,
                   reference_networks=self.reference_networks, target_networks=self.target_networks)
            workunit_manager.makeWorkunitFile()
            if num_worker == 1:
               # Handle single process
               args = getArgs(0, is_terminate=False)
               executeWorker(*args)
            else:
               # Parallel processing
               processes = []
               for process_idx in range(num_worker):
                   proc = mp.Process(target=executeWorker, args=(getArgs(process_idx)))
                   processes.append(proc)
                   proc.start()
               # Wait for the processes to finish
               for process_idx, proc in enumerate(processes):
                   proc.join()
                   printPerformanceInformation(f"parallelFind/time3a, process {process_idx}")
            # Wait for workers to complete
            _print(f"**{num_worker} processes completed.")
            printPerformanceInformation("parallelFind/time4")
            # Merge the results
            merged_checkpoint_result = WorkerCheckpointManager.merge(
                  self.checkpoint_path, num_worker, is_report=is_report)
            printPerformanceInformation(f"parallelFind/time5")
            df = merged_checkpoint_result.merged_checkpoint_manager.recover().full_df
            # Eliminate the checkpoint files
            for worker_idx in range(num_worker):
                worker_checkpoint_path = WorkerCheckpointManager.makeWorkerCheckpointPath(
                    self.checkpoint_path, worker_idx)
                worker_checkpoint_manager = WorkerCheckpointManager(worker_checkpoint_path)
                worker_checkpoint_manager.remove()
        else:
            # No workunits to process
            df = manager.recover()
        # Return the results
        if is_report:
            msg = f"**Done. Processed {len(df)}"
            msg += f" reference networks in {self.checkpoint_path}."
            print(msg)
        return df
    
    @classmethod
    def biomodelsFind(cls,
          reference_network_size:int=15,
          max_num_reference_network:int=-1,
          max_num_target_network:int=-1,
          identity:str=cn.ID_STRONG,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          reference_network_names:List[str]=[],
          is_report:bool=True,
          serialization_dir:str=cn.DATA_DIR,
          reference_serialization_filename:str=REFERENCE_SERIALIZATION_FILENAME,
          target_serialization_filename:str=TARGET_SERIALIZATION_FILENAME,
          checkpoint_path:Optional[str]=None,
          num_worker:int=-1,
          is_no_boundary_network:bool=True,
          skip_networks:Optional[List[str]]=None,
          is_initialize:bool=True)->pd.DataFrame:
        """
        Finds subnets of BioModels models for reference models specified by a maximum size.
        The DataFrame returned includes a reference network, target network pair with null strings for found networks
        so that there is a record of all comparisons made.

        Args:
            reference_network_size (int): Size of networks in Bionetworks that are used as reference network
            max_num_reference_network (int): Maximum number of reference networks to process. If -1, process all
            max_num_target_network (int): Maximum number of target networks to process. If -1, process all
            reference_network_names (List[str]): List of reference network names to process
            is_report (bool): If True, report progress
            serialization_dir (str): Directory for serialization files
            checkpoint_path (Optional[str]): Path for the checkpoint file
            num_worker (int): Number of workers to process the reference networks (-1 is all CPUs)
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
        serializer = ModelSerializer(None, cn.BIOMODELS_SERIALIZATION_PATH)
        collection = serializer.deserialize()
        all_networks = collection.networks
        if skip_networks is not None:
            all_networks = [n for n in all_networks if n.network_name not in skip_networks]
        if is_no_boundary_network:
            all_networks = [n for n in all_networks if not Network.isBoundaryNetwork(n)]
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
          serialization_dir=serialization_dir,
          reference_serialization_filename=reference_serialization_filename,
          target_serialization_filename=target_serialization_filename,
          is_initialize=is_initialize,
          num_worker=num_worker,
          max_num_assignment=max_num_assignment,
          checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    df = ParallelSubnetFinder.biomodelsFind(reference_network_size=10, is_report=True,
          skip_networks=[], num_worker=-1, is_initialize=False)
    df.to_csv(os.path.join(cn.DATA_DIR, "biomodels_subnet.csv"))
    print("Done")