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


class CheckpointManager(object):
 
    def __init__(self, path:str, is_report:bool=True)->None:
        """
        Args:
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
        """
        self.path = path
        self.is_report = is_report
        #
        if self.is_report:
            print(f"CheckpointManager: {self.path}")
            if os.path.exists(self.path):
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
        if os.path.exists(self.path):
            # Read the existing file
            df_existing = pd.read_csv(self.path)
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
            df = pd.read_csv(self.path)
        self._print(f"Recovering a dataframe of length {len(df)} from {self.path}")
        return df