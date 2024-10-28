from utils import State, Instance, DataLoader
from solvers import GreedySolver, RandomGreedySolver, PriorityGreedySolver
from postprocessing import RedundancyElimination
from improvement import SearchStrategy, LocalSearch

import os
from typing import Union, final
import logging
import numpy as np
from tqdm import tqdm
from enum import Enum
from time import perf_counter


class VariableNeighborhoodSearch:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)
    ###
    K_MAX = 3

    def __init__(self, instance: Instance,
                 max_iter_local_search: dict, max_iterations: int,
                 max_iter_neighborhood: dict,
                 logger: logging.Logger = None):

        self.logger = logger if logger else logging.getLogger()
        self.inst = instance
        assert instance.get_state() == State.SOLVED, "Initial solution must be provided"
        assert instance.check_sol_coverage(), "Provided solution is not valid"

        # unpack instance data
        self.mat = self.inst.get_mat()
        self.weights = self.inst.get_weights()
        self.n_rows = self.inst.get_rows()
        self.n_cols = self.inst.get_cols()
        self.cost_optimal = instance.get_best_known_sol_cost()

        # convert solution representation to binary vector
        sol_as_list_of_integers = self.inst.get_sol()
        self.initial_sol = np.zeros(self.n_cols, dtype=int)
        self.initial_sol[sol_as_list_of_integers] = 1
        assert self.is_complete_solution(self.initial_sol)

        # hyper-parameters
        self.max_iterations = max_iterations
        self.max_iter_local_search = max_iter_local_search
        self.max_iter_neighborhood = max_iter_neighborhood

        # statistics
        self.time = None
        self.stats = {
            "step": [],
            "runtime": [],
            "k": [],
            "current-cost": [],
            "candidate-cost": [],
            "local-cost": [],
            "best-cost": [],
            "current-size": [],
            "candidate-size": [],
            "local-size": [],
            "best-size": [],
        }

    # auxiliary methods
    def do_compute_cost(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        cost = self.weights[x == 1].sum()
        return int(cost)

    def do_compute_coverage(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        selected_cols = self.mat[:, x == 1]
        covered_rows = np.bitwise_or.reduce(selected_cols, axis=-1)
        assert covered_rows.size == self.n_rows
        covered_total = covered_rows.sum()
        covered_percent = covered_total / self.n_rows
        return covered_total, covered_percent

    def is_complete_solution(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        covered_total, _ = self.do_compute_coverage(x)
        return covered_total == self.n_rows

    def do_repair(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        y = x.copy()
        for i in range(self.n_cols):
            if self.is_complete_solution(y):
                break
            if y[i]:
                continue
            selected_cols = self.mat[:, y == 1].copy()
            covered_rows = np.bitwise_or.reduce(selected_cols, axis=-1)
            col = self.mat[:, i]
            new_covers = col & ~covered_rows
            if new_covers.sum() > 0:
                y[i] = 1  # include if covers some previously uncovered rows
        assert self.is_complete_solution(y)
        return y

    def do_redundancy_elimination(self, x):
        ranks = np.zeros(self.n_cols)
        redundancy = np.zeros(self.n_cols)
        for i in range(self.n_cols):
            if x[i] == 0:
                ranks[i] = 0
                continue
            y = x.copy()
            y[i] = 0  # remove a column...
            if self.is_complete_solution(y): #...redundant if still complete
                ranks[i] = self.weights[i]
                redundancy[i] = 1
        redundancy = redundancy[ranks.argsort()[::-1]]
        y = x.copy()
        for i in range(self.n_cols):
            if redundancy[i]:
                y[i] = 0
            if not self.is_complete_solution(y):
                y[i] = 1
                break
        return y

    # main methods
    def run(self):
        i = 0
        self.do_start_clock()  # statistics

        # initialization
        current, candidate, best = self.do_initialize()
        cost_current = self.do_compute_cost(current)
        cost_candidate = self.do_compute_cost(candidate)
        cost_best = self.do_compute_cost(best)
        k = 1  # TODO: demote to k=1 after testing
        # pbar = tqdm(total=self.max_iterations, desc="vns", leave=True)
        while i < self.max_iterations:
            j = 0
            while j < self.max_iter_neighborhood.get(k, 1):
                if  k > VariableNeighborhoodSearch.K_MAX:
                    break
                candidate = self.do_shaking(current, k)
                cost_candidate = self.do_compute_cost(candidate)
                local_optimum, cost_local_optimum = self.do_local_search(candidate, cost_candidate, k=k)
                self.do_write_step()  # statistics
                self.do_write_k(k)  # statistics
                self.do_read_clock()  # statistics
                self.do_write_costs(cost_current, cost_candidate, cost_local_optimum, cost_best)  # statistics
                self.do_write_sizes(current, candidate, local_optimum, best)  # statistics
                if cost_local_optimum < cost_current:
                    self.logger.info(f"found improvement (k {k}) -> cost {cost_current},"
                                     f" {cost_best / cost_local_optimum}")

                    current = local_optimum.copy()
                    cost_current = cost_local_optimum
                    break
                else:
                    k += 1
                    continue
            if cost_current < cost_best:
                best = current.copy()
                cost_best = cost_current
            k = 1

            i += 1
            # pbar.set_description(f"vns -> cost {cost_current} ({cost_best} / {self.cost_optimal})")
            # pbar.update()
        self.logger.info(f"reached stop condition after {i} steps "
                         f"(max steps -> {self.max_iterations})")
        assert self.is_complete_solution(best)
        return best, cost_best

    def do_shaking(self, x, k=1):
        # shaking is done with a swap move
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        candidate = x.copy()
        ones_indices = np.where(x == 1)[0]
        zeros_indices = np.where(x == 0)[0]

        if k == 1:  # swap(1, 1)
            one_index = VariableNeighborhoodSearch.RNG.choice(ones_indices)
            candidate[one_index] = 0  # remove 1
            coin = VariableNeighborhoodSearch.RNG.uniform()
            if coin < 0.5:
                zero_index = VariableNeighborhoodSearch.RNG.choice(zeros_indices)
                candidate[zero_index] = 1  # insert 1
        elif k == 2:  # swap(1, 2)
            one_index = VariableNeighborhoodSearch.RNG.choice(ones_indices)
            candidate[one_index] = 0  # remove 1
            coin = VariableNeighborhoodSearch.RNG.uniform()
            if coin < 1/3:
                zero_index = VariableNeighborhoodSearch.RNG.choice(zeros_indices, size=2, replace=False)
                candidate[zero_index] = 1  # insert 2
            elif 1/3 < coin < 2/3:
                zero_index = VariableNeighborhoodSearch.RNG.choice(zeros_indices)
                candidate[zero_index] = 1  # insert 2
            else:
                pass
        elif k == 3:  # swap(2, 2)
            coin = VariableNeighborhoodSearch.RNG.uniform()
            if coin < 0.5:
                one_index = VariableNeighborhoodSearch.RNG.choice(ones_indices, size=2, replace=False)
                candidate[one_index] = 0  # remove 2
            else:
                one_index = VariableNeighborhoodSearch.RNG.choice(ones_indices)
                candidate[one_index] = 0  # remove 1

            coin = VariableNeighborhoodSearch.RNG.uniform()
            if coin < 1/3:
                zero_index = VariableNeighborhoodSearch.RNG.choice(zeros_indices, size=2, replace=False)
                candidate[zero_index] = 1  # insert 2
            elif 1/3 < coin < 2/3:
                zero_index = VariableNeighborhoodSearch.RNG.choice(zeros_indices)
                candidate[zero_index] = 1  # insert 2
            else:
                pass

        if self.is_complete_solution(candidate):
            candidate = self.do_redundancy_elimination(candidate)
            return candidate
        else:
            candidate = self.do_repair(candidate)
            candidate = self.do_redundancy_elimination(candidate)
            return candidate

    def do_local_search(self, x, cost_x, k):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols

        local_optimum = x.copy()
        cost_local_optimum = cost_x
        total_iter = self.max_iter_local_search.get(k, 1_000)
        pbar = tqdm(total=total_iter, desc="local search", leave=False)
        for i in range(total_iter):  # move: 1-flip
            y = x.copy()
            flip_indices = VariableNeighborhoodSearch.RNG.integers(0, self.n_cols, size=1)
            y[flip_indices] = 1 - y[flip_indices]
            if not self.is_complete_solution(y):
                y = self.do_repair(y)
            y = self.do_redundancy_elimination(y)
            cost_y = self.do_compute_cost(y)
            # self.logger.info(f"local search {k} -> flip {flip_indices[0]}, cost {cost_y}")
            pbar.set_description(f"local search -> {cost_y} ({self.cost_optimal})")
            pbar.update()
            if cost_y < cost_x:
                local_optimum = y.copy()
                cost_local_optimum = cost_y
        pbar.close()
        return local_optimum, cost_local_optimum

    def do_initialize(self):
        current = self.initial_sol.copy()
        candidate = self.initial_sol.copy()
        best = self.initial_sol.copy()
        return current, candidate, best

    def get_stats(self):
        return self.stats

    # Statistics
    def do_start_clock(self):
        self.time = perf_counter()

    def do_read_clock(self):
        now = perf_counter()
        self.stats["runtime"].append(now - self.time)
        self.time = now

    def do_write_step(self):
        if len(self.stats["step"]) < 1:
            self.stats["step"].append(0)
        else:
            last_step = self.stats["step"][-1]
            self.stats["step"].append(last_step + 1)

    def do_write_k(self, k):
        self.stats["k"].append(k)

    def do_write_costs(self, current, candidate, local, best):
        self.stats["current-cost"].append(current)
        self.stats["candidate-cost"].append(candidate)
        self.stats["local-cost"].append(local)
        self.stats["best-cost"].append(best)

    def do_write_sizes(self, current, candidate, local, best):
        self.stats["current-size"].append(len(current))
        self.stats["candidate-size"].append(len(candidate))
        self.stats["local-size"].append(len(local))
        self.stats["best-size"].append(len(best))

    def configure_logger(self, log_file_path):
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_cost_optimal(self):
        return self.cost_optimal

    def set_hyper_parameters(self, **kwargs):
        if max_iterations := kwargs.get("max_iterations", None):
            self.max_iterations = max_iterations
        if max_iter_local_search := kwargs.get("max_iter_local_search", None):
            self.max_iter_local_search = max_iter_local_search

if __name__ == "__main__":
    dl = DataLoader()
    inst_name = "42"
    inst = dl.load_instance(inst_name)

    OUTPUT_DIR = "../output"
    if not os.path.exists(path := os.path.join(OUTPUT_DIR, "vns-test-run")):
        os.mkdir(path)
    log_path = os.path.join(OUTPUT_DIR, "vns-test-run", f"{inst_name}.log")

    solver = GreedySolver(instance=inst)
    solver.configure_logger(path=log_path)
    sol_greedy, cost_greedy = solver.greedy_heuristic()

    # reset instance
    inst = dl.load_instance(inst_name)
    inst.set_sol(list(sol_greedy))
    inst.set_state(State.SOLVED)

    # run the VNS
    N = inst.get_cols()
    frac = 0.8
    max_iter = 1_000

    max_iter_neighborhood = {
        1: 10,
        2: 100,
        3: 100,
    }
    max_iter_ls = {
        1: int(1.5 * N),
        2: int(1.5 * N),
        3: int(1.5 * N),
    }
    print(f"max_iter_ls: {max_iter_ls}")
    vns = VariableNeighborhoodSearch(instance=inst,
                                     max_iter_local_search=max_iter_ls, max_iterations=max_iter,
                                     max_iter_neighborhood=max_iter_neighborhood
                                     )
    vns.configure_logger(log_path)

    best, cost_best = vns.run()
    cost_optimal = vns.get_cost_optimal()
    is_hit = cost_optimal == cost_best
    stats = vns.get_stats()

    import matplotlib.pyplot as plt

    time = stats["step"]
    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=2.4, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.8, ls="-", alpha=0.9)
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=1.2, ls="-", alpha=0.7)
    plt.plot(time, stats["local-cost"], label="local-cost",
             color="tab:orange", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(cost_optimal, color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.bar(time, stats["k"],
              color="tab:gray", alpha=0.3, zorder=-10)
    plt.ylabel("k (neighborhood order)")

    plt.title(f"vns - {inst_name} \n "
              f"hit: {is_hit} -> best: {cost_best}, opt: {cost_optimal}")
    plt.show()
