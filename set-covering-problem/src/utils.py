import os
import numpy as np
from glob import glob
from enum import Enum


class State(Enum):
    NOT_LOADED = 0
    LOADED = 1
    SOLVED = 2

class Instance:
  
    def __init__(self, n_cols, n_rows):
        self.state = State.NOT_LOADED 

        self.n_cols = n_cols  # cols = subsets
        self.n_rows = n_rows  # rows = elements
        self.total_covered_by = []  # how many subsets cover each row
        self.covered_by = []  # which subsets cover each row
        
        self.weights = np.zeros(self.n_cols)  # costs
        self.mat = np.zeros((self.n_rows, self.n_cols), dtype=bool)  # boolean array representation

        self.sol = []  # which subsets are included in solution
        self.sol_coverage = np.zeros(self.n_rows)  # how many times solution covers each element
    

    def update_sol_coverage(self):
        self.sol_coverage = self.mat[:, self.sol].sum(axis=1).flatten()

    def check_sol_coverage(self):
        self.update_sol_coverage()
        return np.all(self.sol_coverage != 0)

    def get_coverage_percent(self):
        self.update_sol_coverage()
        return (self.sol_coverage > 0).sum() / self.n_rows

    def get_coverage_per_element(self):
        self.update_sol_coverage()
        return self.sol_coverage

    def get_sol_cost(self):
        return self.weights[self.sol].sum()

    def set_state(self, state: State) -> None:
        self.state = state

    def set_weights(self, weights):
        self.weights[:] = weights

    def set_row_coverage(self, total_covered_by, covered_by):
        self.total_covered_by.extend(total_covered_by)
        self.covered_by.extend(covered_by)

    def set_mat(self, mat):
        self.mat = mat

    def get_cols(self):
        return self.n_cols

    def get_rows(self):
        return self.n_rows

    def get_sol(self):
        return self.sol

    def get_mat(self):
        return self.mat

    def increment_sol(self, i):
        self.sol.append(i)

    def prune_sol(self, i):
        self.sol.remove(i)

    def get_weights(self):
        return self.weights

    def get_state(self):
        return self.state

class DataLoader:
    DATA_DIR = "../data"

    def __init__(self):
        """
        The format of all of these 80 data files is:
        number of rows (m), number of columns (n)
        the cost of each column c(j),j=1,...,n
        for each row i (i=1,...,m): the number of columns which cover
        row i followed by a list of the columns which cover row i
        """
        self.available_instances = self.list_instances()
        return

    @staticmethod
    def list_instances():
        names = [
            os.path.basename(name)[3:-4] for name in glob(os.path.join(
                DataLoader.DATA_DIR, "*.txt"))
        ]
        return sorted(names)

    def load_instance(self, name):
        if name not in self.available_instances:
            return

        file = os.path.join(DataLoader.DATA_DIR, f"scp{name}.txt")
        with open(file, "r") as f:
            header = self.process_line(f)
            n_rows, n_cols = header[0], header[1]
            instance = Instance(n_cols=n_cols, n_rows=n_rows)

            weights = self.load_weights(f, n_cols=n_cols)
            instance.set_weights(weights)

            total_covered_by, covered_by = self.load_row_coverage(f, n_rows)
            instance.set_row_coverage(
                total_covered_by=total_covered_by, covered_by=covered_by
            )

            mat = self.compute_mat(n_cols=n_cols, n_rows=n_rows,
                                   total_covered_by=total_covered_by,
                                   covered_by=covered_by)
            instance.set_mat(mat)

            instance.set_state(State.LOADED)
            return instance

    @staticmethod
    def process_line(f):
        line = f.readline()
        if not line:
            return None
        return list(map(int, line.strip().split()))

    def load_weights(self, f, n_cols):
        weights = []
        while len(weights) < n_cols:
            w = self.process_line(f)
            weights.extend(w)
        assert len(weights) == n_cols, f"Expected {n_cols} weights, but got {len(weights)}"
        return weights

    def load_row_coverage(self, f, n_rows):
        total_covered_by = []
        covered_by = []

        while len(covered_by) < n_rows:
            total = self.process_line(f)

            if total is None:
                break  # end of file

            if len(total) == 1:
                total_covered_by.extend(total)
                temp = []
                while len(temp) < total[0]:
                    temp.extend(self.process_line(f))
                covered_by.append(temp)

        return total_covered_by, covered_by

    @staticmethod
    def compute_mat(n_cols, n_rows, total_covered_by, covered_by):
        mat = np.zeros((n_rows, n_cols), dtype=bool)
        for row in range(n_rows):
            row_covered_by = covered_by[row]
            for col in row_covered_by:
                mat[row, col - 1] = 1

        mat_sum = mat.sum(axis=-1)
        tcb = np.array(total_covered_by)

        assert np.all(mat.sum(axis=-1) == np.array(total_covered_by))
        return mat

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("42")
    print(inst.n_rows, inst.n_cols)
    print(len(inst.weights))
    print(len(inst.total_covered_by))
    print(len(inst.covered_by))
    print(inst.mat.shape)

    inst.sol = [0, 1, 2]
    print(inst.get_coverage_percent())
    print(inst.check_sol_coverage())
    print(inst.get_sol_cost())

    inst.sol = [i for i in range(inst.n_cols)]
    print(inst.get_coverage_percent())
    print(inst.check_sol_coverage())
    print(inst.get_sol_cost())