'''Timing for performance analysis'''

import os
import time
import psutil # type: ignore
from typing import List, Tuple


def getMemoryUsage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss # in bytes


class PerformanceMonitor(object):

    def __init__(self, name:str, is_enabled:bool=True):
        self.is_enabled = is_enabled
        self.name = name
        self.memory_usages:list = []
        self.elapsed_times:list = []
        self.base_time = time.time()
        self.names:list = [name + ": Start"]
        if self.is_enabled:
            print(f"\n**** {name}")

    def add(self, name:str):
        if not self.is_enabled:
            return
        self.names.append(name)
        elapsed_time = time.time() - self.base_time
        self.elapsed_times.append(elapsed_time)
        memory_usage = getMemoryUsage()/1e6  # in MB
        self.memory_usages.append(memory_usage)
        print(f"{self.names[-2]} - {name}: {elapsed_time}, {memory_usage}")
        self.base_time = time.time()