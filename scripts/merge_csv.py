'''Merges multiple CSV files into a single file.'''

import argparse
import os
import pandas as pd # type: ignore
import sys
from typing import Optional

def merge(directory:str=".", file_str:str="", output_path:Optional[str]=None):
    """
    Merges multiple CSV files into a single file.
    Args:
        directory (str): Path to the directory containing the CSV files. If None, use current directory.
        file_str (str): Filename contains this string.
        output_path (str): Path to the output CSV file. If None, use standard output
    """
    # Find the files
    ffiles = [f for f in os.listdir(directory) if (file_str in f) and f.endswith(".csv")]
    if len(ffiles) == 0:
        raise FileNotFoundError(f"No files found in {directory} with the string {file_str}")
    # Read the files
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in ffiles]
    # Merge the files
    df = pd.concat(dfs)
    # Write the output
    if output_path is None:
        df.to_csv(sys.stdout, index=False)
    else:
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into a single file.")
    parser.add_argument("--directory", type=str, default=".", help="Path to the directory containing the CSV files.")
    parser.add_argument("--file_str", type=str, default="", help="Filename contains this string.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output CSV file.")
    args = parser.parse_args()
    merge(args.directory, args.file_str, args.output_path)