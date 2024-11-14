'''Does timings for calculating comparison of matrices.'''
"""
Preliminary numbers on 1e5 matrices of size 5x10: 12 microseconds per comparison, 50% identical.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # type: ignore
import time
from typing import Tuple, List
import seaborn as sns # type: ignore

def timeMatrixComparison(count:int, nrow:int, ncol:int)->Tuple[float, float]:
    """Counts the number of identical matrices and times it.

    Args:
        count (int):
        nrow (int):
        ncol (int):
    Returns:
        float: Average time for comparison (msec)
        float: fraction of identical matrices.
    """
    randoms = np.random.rand(count)
    num_identical = 0
    total_time:float = 0
    for idx in range(count):
        arr1 = np.random.randint(-1, 2, (nrow, ncol))
        if randoms[idx] < 0.5:
            arr2 = arr1
        else:
            arr2 = np.random.randint(-1, 2, (nrow, ncol))
        start = time.time()
        result = np.allclose(arr1, arr2)
        end = time.time()
        total_time += end - start
        if result:
            num_identical += 1
    #
    avg_mstime = 1000*total_time/count
    fraction_identical = num_identical/count
    return avg_mstime, fraction_identical

def countComparisons(nrows:np.ndarray, ncols:np.ndarray)->np.ndarray:
    """Counts the number of comparisons for a given matrix size.

    Args:
        nrow (list-int): Number of rows
        ncol (list-int): Number of columns
    Returns:
        float: log10 of the number of comparisons
    """
    def sumOfLog(num:int)->float:
        """Sum of log10 of integers from 1 to num."""
        return sum([np.log10(i) for i in range(1, num+1)])
    return np.array([sumOfLog(r) + sumOfLog(c) for r, c in zip(nrows, ncols)])

def countSubnetComparisons(reference_size_arr:np.ndarray, target_size_arr:np.ndarray)->np.ndarray:
    """Counts the number of comparisons for a square matrix of specified size.

    Args:
        nrow (list-int): Number of rows
        ncol (list-int): Number of columns
    Returns:
        float: log10 of the number of comparisons
    """
    def sumOfLog(num:int, start_num:int=1)->float:
        """Sum of log10 of integers from 1 to num."""
        return sum([np.log10(i) for i in range(start_num, num+1)])
    def logChoose(big_num:int, small_num:int)->float:
        """Log of big_num choose small_num."""
        numr = sumOfLog(big_num, start_num=small_num+1)
        denom = sumOfLog(big_num - small_num)
        return numr - denom
    # Calculate the number of assignments for rows (columns)
    arr = np.array([(logChoose(t, r) + sumOfLog(r))  if r <= t else np.nan
          for r, t in zip(reference_size_arr, target_size_arr)])
    # Double this to get both rows and columns
    return 2*arr

def heatmapCountComparisons(nrows:np.ndarray, ncols:np.ndarray)->None:
    """Plots the number of comparisons for a given matrix size.

    Args:
        nrow (list-int): Number of rows
        ncol (list-int): Number of columns
    """
    MICROSECONDS_PER_COMPARISON = 12
    data = countComparisons(nrows, ncols)
    plot_df = pd.DataFrame({'nrows': nrows, 'ncols': ncols, 'data': data})
    plot_df = plot_df.pivot(index="nrows", columns="ncols", values="data")
    plot_df = plot_df.sort_index(ascending=False)
    # Translate to hours
    plot_df +=  np.log10(MICROSECONDS_PER_COMPARISON)
    plot_df -=  np.log10(1e6*60*60)
    ax = sns.heatmap(plot_df, cmap="seismic", vmin=-18, vmax=18,
        cbar_kws={'label': 'log10(hours)'})
    ax.set_xlabel("Number of reactions")
    ax.set_ylabel("Number of species")
    plt.show()

def heatmapAssignmentComparisons(reference_size_arr:np.ndarray, target_size_arr:np.ndarray)->None:
    """Plots the number of comparisons for subset assignment.

    Args:
        reference_size_arr (list-int): Size of the reference network
        target_size_arr (list-int): Size of the target network
    """
    data = countSubnetComparisons(reference_size_arr, target_size_arr)
    plot_df = pd.DataFrame({'reference_size_arr': reference_size_arr, 'target_size_arr': target_size_arr, 'data': data})
    plot_df = plot_df.pivot(index="reference_size_arr", columns="target_size_arr", values="data")
    plot_df = plot_df.transpose()
    plot_df = plot_df.sort_index(ascending=False)
    # Translate to hours
    ax = sns.heatmap(plot_df, cmap="Reds", vmin=-1, vmax=25, annot=True,
        cbar_kws={'label': 'log10(# assignments)'})
    ax.set_xlabel("reference")
    ax.set_ylabel("target")
    plt.show()

if __name__ == '__main__':
    if False:
        size = 10
        arr =  2.0*np.array(range(1, size + 1))
        sub_nrows = arr.astype(int)
        nrows = [np.repeat(n, size) for n in sub_nrows]
        nrows = np.concatenate(nrows)
        ncols = np.concatenate([sub_nrows]*size)
        ncols = ncols.flatten()
        ncols = ncols.astype(int)
        nrows = np.flip(nrows)   # type: ignore
        heatmapCountComparisons(nrows, ncols)  # type: ignore
    if False:
        # testCountSUbsetComparisons
        reference_size_arr = [2, 4, 6]
        target_size_arr = [10, 20, 100]
        result = countSubnetComparisons(reference_size_arr, target_size_arr)
    if True:
        reference_sizes = [2*n for n in range(1, 11)]
        target_sizes = [5*n for n in range(1, 21)]
        reference_size_arr = np.vstack(reference_sizes*len(target_sizes)).flatten()
        target_size_arr = np.repeat(target_sizes, len(reference_sizes))
        heatmapAssignmentComparisons(reference_size_arr, target_size_arr)