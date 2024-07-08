import analysis.constants as cn  # type: ignore
from analysis.summary_statistics import SummaryStatistics  # type: ignore
import sirn.constants as cnn  # type: ignore

import os
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore


def plotMetricComparison():
    for metric in [cn.COL_PROCESSING_TIME, cn.COL_IS_INDETERMINATE, cn.COL_NUM_PERM]:
        SummaryStatistics.plotMetricComparison(metric)


if __name__ == '__main__':
    plotMetricComparison()