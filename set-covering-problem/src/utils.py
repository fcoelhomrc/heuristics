import os
import numpy as np
from glob import glob
from enum import Enum

DATA_DIR = "../data"

class State(Enum):
    NOT_LOADED = 0
    LOADED = 1
    SOLVED = 2

class Instance:
  
    def __init__(self):
        self.state = State.NOT_LOADED 

        self.ncols = None  # cols = subsets
        self.nrows = None  # rows = elements
        self.total_covered_by = []  # how many subsets cover each row
        self.covered_by = []  # which subsets cover each row
        
        self.weights = np.zeros(self.ncols)  # costs
        self.mat = np.zeros((self.nrows, self.ncols))  # boolean array representation

        self.sol = []  # which subsets are included in solution
        self.sol_coverage = np.zeros(self.nrows)  # how many times solution covers each element

    

    def update_sol_coverage(self):
        self.sol_coverage = self.mat[:, self.sol].sum(axis=0).flatten()

    def check_sol_coverage(self):
        self.update_sol_coverage()
        return np.all(self.sol_coverage != 0)

    def get_coverage_percent(self):
        self.update_sol_coverage()
        return (self.sol_coverage > 0).sum() / self.nrows

    def get_sol_cost(self):
        return self.weights[self.sol].sum()

    def set_state(self, state: State) -> None:
        self.state = state 

class DataLoader:

    def __init__(self):
        return

    def list_instances():
        return

    def load_instance(name):
        instance = Instance()
        #TODO: load data...
        instance.set_state(State.LOADED)
        return instance


