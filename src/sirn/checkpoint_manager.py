'''Handles recovery and restarting long running tasks.'''

"""
    This module provides a class that manages the checkpointing of long running tasks.
    The CheckpointManager is constructued by the client providing a path to a CSV file
    (which may be non-existent).
    
    For each checkpoint, the client provides a DataFrame with the specified column. CheckpointManager
    updates the CSV file with the new data.

    For recovery, CheckpointManager reads the CSV file and returns a dataframe of the CSV.
 """

import numpy as np
import os  # type: ignore
import pandas as pd  # type: ignore
from typing import List


class CheckpointManager(object):
 
    def __init__(self, path:str, is_report:bool=True, is_initialize:bool=False)->None:
        """
        Args:
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
            is_initialize(bool): If True, deletes any existing checkpoint file
        """
        self.path = path
        self.is_report = is_report
        #
        self._print(f"CheckpointManager: {self.path}")
        if os.path.exists(self.path):
            if is_initialize:
                os.remove(self.path)
                self._print(f"Deleted: {self.path}")
            else:
                self._print(f"Recovering from: {self.path}")
        else:
            self._print(f"Creating: {self.path}")

    def _print(self, msg:str)->None:
        if self.is_report:
            print(f"***{msg}")

    def checkpoint(self, df:pd.DataFrame)->None:
        """
        Checkpoints a DataFrame.
        """
        self._print(f"Checkpointing a dataframe of length {len(df)} to {self.path}")
        df.to_csv(self.path, index=False)

    def recover(self)->pd.DataFrame:
        """
        Recovers a previously saved DataFrame.

        Returns:
            np.ndarray: List of processed tasks
        """
        if not os.path.exists(self.path):
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(self.path)
            except:
                df = pd.DataFrame()
        self._print(f"Recovering a dataframe of length {len(df)} from {self.path}")
        return df
    
    def mergeCheckpoint(self, checkpoint_managers:List['CheckpointManager'])->int:
        """
        Merges the checkpoints from the checkpoint managers.

        Args:
            List (CheckpointManager): List of CheckpointManager objects

        Returns:
            int: Number of rows in the merged dataframe
        """
        dfs = [cm.recover() for cm in checkpoint_managers]
        df = pd.concat(dfs, ignore_index=False)
        self.checkpoint(df)
        return len(df)
    
    def remove(self):
        """Remove the checkpoint file."""
        if os.path.exists(self.path):
            os.remove(self.path)
            self._print(f"Deleted: {self.path}")