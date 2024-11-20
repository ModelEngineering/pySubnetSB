'''Worker for running SubnetFinder in parallel.'''

import sirn.constants as cn # type: ignore
from sirn.subnet_finder import SubnetFinder # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
from sirn.worker_checkpoint_manager import WorkerCheckpointManager  # type: ignore

import multiprocessing as mp
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Tuple, Optional


def executeTask(task_idx:int, queue, total_task:int, checkpoint_path:str,
        reference_serialization_path:str, target_serialization_path:str,
        identity:str=cn.ID_STRONG, is_report:bool=True,
        is_initialize:bool=True, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->pd.DataFrame:
    """
    Processes pairs of reference and target networks. The reference networks are processed
    based on the index provided in the queue. A task has its own checkpoint file.

    Args:
        task_idx (int): Index of the task
        queue (Queue): Queue of Workunit
        total_task (int): Total number of tasks (including this task)
        checkpoint_path (str): Output path for the consoldated checkpoint files
        reference_serialization_path (str): Path to the serialized reference networks
        target_serialization_path (str): Path to the serialized target networks
        identity (str): Identity type
        is_report (float): Task 0 reports progress
        is_initialize (bool): If True, initialize the checkpoint
        max_num_assignment (int): Maximum number of assignment pairs

    Returns:
        pd.DataFrame: Table of matching models
            reference_model (str): Reference model name
            target_model (str): Target model name
            reference_network (str): may be the null string
            target_network (str): may be the null string
    """
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
    # Calculate the number of processes per task
    num_process = int(mp.cpu_count() / total_task)
    # Get the reference and target models
    reference_networks = getNetworks(reference_serialization_path)
    target_networks = getNetworks(target_serialization_path)
    # Set up the checkpoint manager for this task
    is_report_task = is_report if task_idx == 0 else False
    task_checkpoint_path = WorkerCheckpointManager.makeTaskPath(checkpoint_path, task_idx)
    task_checkpoint_manager = WorkerCheckpointManager(task_checkpoint_path, is_report=is_report_task,
          is_initialize=is_initialize)
    # Get the processed reference networks
    full_task_df = task_checkpoint_manager.recover().full_df
    # Process the unprocessed reference models in batches
    msg = f"**Task start {task_idx} with"
    msg += " {len(reference_networks) - len(processed_reference_networks)} networks."
    if is_report_task:
        print(msg)
    prior_merged_checkpoint_result = None
    while True:
        workunit = queue.get()
        if workunit.is_done:
            queue.task_done()
            break
        # Extract the networks for this batch as specified by -1 being a wildcard
        if workunit.reference_idx < 0:
            batch_reference_networks = reference_networks
        else:
            batch_reference_networks = [reference_networks[workunit.reference_idx]]
        if workunit.target_idx < 0:
            batch_target_networks = target_networks
        else:
            batch_target_networks = [target_networks[workunit.target_idx]]
        # Construct the notification
        if len(batch_reference_networks) == 1:
            msg = f"**Task {task_idx} processing reference network {batch_reference_networks[0].network_name}."
        elif len(batch_target_networks) == 1:
            msg = f"**Task {task_idx} processing target network {batch_target_networks[0].network_name}."
        else:
            msg = f"**Task {task_idx} processing {len(batch_reference_networks)} reference networks."
        if is_report_task:
            print(msg)
        # Finding the Subnets
        finder = SubnetFinder(batch_reference_networks, batch_target_networks, identity=identity,
                num_process=num_process)
        incremental_df = finder.find(is_report=is_report_task, max_num_assignment=max_num_assignment)
        full_task_df = pd.concat([full_task_df, incremental_df], ignore_index=True)
        task_checkpoint_manager.checkpoint(full_task_df)
        # Update the processed networks
        merged_checkpoint_result = WorkerCheckpointManager.merge(checkpoint_path, total_task,
              merged_checkpoint_result=prior_merged_checkpoint_result, is_report=is_report_task)
        prior_merged_checkpoint_result = merged_checkpoint_result
        if is_report_task:
            print(f"**Processed {merged_checkpoint_result.num_reference_network} work units.")
    #
    if is_report_task:
        print(f"**Task {task_idx} done.")
    return full_task_df
    
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