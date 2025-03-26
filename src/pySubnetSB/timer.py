'''Timing for performance analysis'''

import time

class Timer(object):

    def __init__(self, name:str, is_enabled:bool=True):
        self.is_enabled = is_enabled
        self.name = name
        self.base_time = time.time()
        self.dct:dict = {}

    def add(self, name:str):
        if not self.is_enabled:
            return
        print(f"Adding {name}")
        self.dct[name] = time.time() - self.base_time
        self.base_time = time.time()

    def report(self):
        if not self.is_enabled:
            return
        print(f"***{self.name}***")
        for key, value in self.dct.items():
            print(f"  {key}: {value}")
        print("************")
