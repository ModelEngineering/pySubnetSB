'''Worker runs in a single process, reads workunits, calls SubnetFinder, and writes a results checkpoint file.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.subnet_finder import SubnetFinder # type: ignore
from pySubnetSB.model_serializer import ModelSerializer # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.worker_checkpoint_manager import WorkerCheckpointManager  # type: ignore
from pySubnetSB.subnet_finder_workunit_manager import SubnetFinderWorkunitManager # type: ignore

import pandas as pd  # type: ignore
import sys
import time
from typing import List, Optional

REPORT_INTERVAL = 1000


def executeWorker(worker_idx:int, workunit_path, num_worker:int, checkpoint_path:str,
        reference_serialization_path:str, target_serialization_path:str,
        identity:str=cn.ID_STRONG, is_report:bool=True,
        is_initialize:bool=True,
        max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
        checkpoint_interval:int=10,
        is_allow_exit:bool=True)->None:
    """
    Processes pairs of reference and target networks. Networks are processed
    based on the index provided in the queue. A worker creates checkpoint file that is a
    serialization of the DataFrame structured as described in SubnetFinder
    (cn.FINDER_DATAFRAME_COLUMNS).
    The checkpoint file is used to recover the state of the worker, including its completion.

    Args:
        worker_idx (int): Index of the worker
        workunit_path (str): Path to the workunit file
        num_worker (int): Total number of workers
        checkpoint_path (str): Output path for the consoldated checkpoint files
        reference_serialization_path (str): Path to the serialized reference networks
        target_serialization_path (str): Path to the serialized target networks
        identity (str): Identity type
        is_report (float): Worker 0 reports progress
        is_initialize (bool): If True, initialize the checkpoint
        max_num_assignment (int): Maximum number of assignment pairs
        checkpoint_interval (int): Interval for checkpointing
        is_allow_exit (bool): If True, exit after completion (faster handling of queue merge for parallel)
    """
    LAST_TIME = "last_time"
    CHECKPOINT_TIME = 5*60  # 5 minutes
    global_dct = {LAST_TIME: time.time()}
    #####
    def printPerformanceInformation(msg:str):
        # Print performance information
        if False:
            last_time = global_dct[LAST_TIME]
            now_time = time.time()
            duration = now_time - last_time
            new_msg = f"**{msg}: {duration} seconds."
            print(new_msg, flush=True)
            global_dct[LAST_TIME] = now_time
    #####
    def getNetworks(serialization_path:str)->List[Network]:
        """
        Obtains the networks from a serialization file.

        Args:
            serialization_path (str): Path to the serialization file

        Returns:
            List[Network]: List of networks
        """
        serializer = ModelSerializer(None, serialization_path)
        collection = serializer.deserialize()
        return collection.networks
    #####
    is_report_worker = is_report if worker_idx == 0 else False
    # Set up the checkpoint manager for this worker
    worker_checkpoint_path = WorkerCheckpointManager.makeWorkerCheckpointPath(checkpoint_path,
          worker_idx)
    worker_checkpoint_manager = WorkerCheckpointManager(worker_checkpoint_path,
          is_report=is_report_worker, is_initialize=is_initialize)
    full_worker_df = worker_checkpoint_manager.recover().full_df
    # Get the workunits
    if len(full_worker_df) == 0:
        processed_reference_networks = []
        processed_target_networks = []
    else:
        processed_reference_networks  = full_worker_df[cn.FINDER_REFERENCE_NAME]
        processed_target_networks = full_worker_df[cn.FINDER_TARGET_NAME]
    workunit_manager = SubnetFinderWorkunitManager(workunit_path, num_worker=num_worker)
    workunits = workunit_manager.getWorkunits(worker_idx,
          processed_reference_networks=processed_reference_networks,
          processed_target_networks=processed_target_networks)
    # Get the reference and target models
    reference_networks = getNetworks(reference_serialization_path)
    target_networks = getNetworks(target_serialization_path)
    # Process the unprocessed reference models in batches
    msg = f"**Worker start {worker_idx} with"
    msg += " {len(reference_networks) - len(processed_reference_networks)} networks."
    if is_report_worker:
        print(msg, flush=True)
    prior_merged_checkpoint_result = None
    printPerformanceInformation(f"**Worker {worker_idx} start.")
    workunit_idx = 0
    last_checkpoint_time = time.process_time()
    for idx, workunit in enumerate(workunits):
        is_report_loop = is_report_worker and workunit_idx % REPORT_INTERVAL == 0
        finder = SubnetFinder.makeFromCombinations([reference_networks[workunit.reference_idx]],
              [target_networks[workunit.target_idx]],
              identity=identity,
              num_process=1)
        incremental_df = finder.find(is_report=is_report_loop, max_num_assignment=max_num_assignment)
        full_worker_df = pd.concat([full_worker_df, incremental_df], ignore_index=True)
        if time.process_time() - last_checkpoint_time > CHECKPOINT_TIME:
            worker_checkpoint_manager.checkpoint(full_worker_df)
            last_checkpoint_time = time.process_time()
        # Update the processed networks
        if worker_idx == 0 and workunit_idx % checkpoint_interval == 0:
            merged_checkpoint_result = WorkerCheckpointManager.merge(checkpoint_path, num_worker,
                merged_checkpoint_result=prior_merged_checkpoint_result, is_report=is_report_worker)
            prior_merged_checkpoint_result = merged_checkpoint_result
        if is_report_loop:
            print(f"**Processed {idx} work units.", flush=True)
        workunit_idx += 1
    # Do final checkpoint for worker
    worker_checkpoint_manager.checkpoint(full_worker_df)
    printPerformanceInformation(f"Completed process {worker_idx}.")
    #
    if is_report_worker:
        print(f"**Worker {worker_idx} done.", flush=True)
    if is_allow_exit:
        sys.exit(0)
    
#############################################################
class Workunit(object):
    def __init__(self, reference_idx:int=-1, target_idx:int=-1, is_done:bool=False):
        """
        Args:
            reference_idx (int, optional): index of reference network. Defaults to -1 means all.
            target_idx (int, optional): index of target network. Defaults to -1 means all.
            is_done (bool, optional): No more work.
        """
        self.reference_idx = reference_idx
        self.target_idx = target_idx
        self.is_done = is_done

    def __repr__(self):
        return f"Workunit(ref={self.reference_idx}, tgt={self.target_idx}, is_done={self.is_done})"
    
    def __eq__(self, other):
        return (self.reference_idx == other.reference_idx) and (self.target_idx == other.target_idx)
    
    @classmethod
    def addMultipleWorkunits(cls, queue, reference_idxs:Optional[List[int]]=None,
          target_idxs:Optional[List[int]]=None)->None:
        """
        Adds a list of integer workunits to the queue that is the cross product
        of the reference and target indexes.

        Args:
            queue (Queue): Queue to add the workunits
            reference_idxs (Optional[List[int]]): List of reference indexes
            target_idxs (Optional[List[int]]): List of target indexes
        """
        if reference_idxs is None:
            reference_idxs = [-1]
        if target_idxs is None:
            target_idxs = [-1]
        workunits = [cls(reference_idx=ref_idx, target_idx=tgt_idx)
              for ref_idx in reference_idxs for tgt_idx in target_idxs]
        [queue.put(w) for w in workunits]