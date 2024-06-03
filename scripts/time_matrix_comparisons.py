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

def heatmapCountComparisons(nrows:np.ndarray, ncols:np.ndarray)->None:
    """Plots the number of comparisons for a given matrix size.

    Args:
        nrow (list-int): Number of rows
        ncol (list-int): Number of columns
    """
    data = countComparisons(nrows, ncols)
    plot_df = pd.DataFrame({'nrows': nrows, 'ncols': ncols, 'data': data})
    plot_df = plot_df.pivot(index="nrows", columns="ncols", values="data")
    plot_df = plot_df.sort_index(ascending=False)
    # Translate to hours
    adjust = np.log10(12*1e-6*60*60)
    plot_df += adjust
    import pdb; pdb.set_trace()
    ax = sns.heatmap(plot_df, cmap="seismic", vmin=-18, vmax=18,
        cbar_kws={'label': 'log10(hours)'})
    ax.set_xlabel("Number of reactions")
    ax.set_ylabel("Number of species")
    plt.show()

if __name__ == '__main__':
    #print(timeMatrixComparison(100000, 5, 10))
    nrows =  0.2*np.array([5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 80, 80, 80, 80, 80])
    nrows = nrows.astype(int)
    ncols =  0.2*np.array([5, 10, 20, 40, 80])
    ncols = np.concatenate([ncols]*5)
    ncols = ncols.astype(int)
    nrows = np.sort(nrows)
    nrows = np.flip(nrows)
    #print(countComparisons(nrows, ncols))
    heatmapCountComparisons(nrows, ncols)