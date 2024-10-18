import os
from pytz.reference import Local

from utils import DataLoader, Instance, State
import numpy as np
import logging
from time import perf_counter
from solvers import *
from enum import Enum
from typing import Union
from tqdm import tqdm


class SearchStrategy(Enum):
    FIRST = 0
    BEST = 1


class LocalSearch:

    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self,
                 heuristic: Union[GreedySolver, RandomGreedySolver, PriorityGreedySolver],
                 strategy: SearchStrategy = SearchStrategy.FIRST):

        # assert heuristic.inst.get_state() == State.SOLVED

        self.solver = heuristic
        self.strategy = strategy

        self.logger = self.solver.logger
        self.inst = self.solver.inst # Reduce verbosity in this implementation
        self.initial_sol = self.inst.get_sol()

        self.max_iter = 1000
        self.max_iter_per_neighborhood = 100

        self.incumbent_sol = self.initial_sol.copy()
        self.neighbor = self.incumbent_sol.copy()
        self.best_neighbor = self.neighbor.copy()

        self.improvement_step_elapsed_time = None
        self.run_data = {
            "valid-neighbor-cost": []
        }

    def do_n_insert(self, n=1):
        assert isinstance(n, int)

        candidates = np.arange(self.inst.get_cols(), dtype=int)
        to_insert =  LocalSearch.RNG.choice(candidates, size=n)
        for candidate in to_insert:
            if candidate in self.incumbent_sol:
                continue
            self.neighbor.append(candidate)

    def do_n_remove(self, n=1):
        assert isinstance(n, int)

        to_remove =  LocalSearch.RNG.choice(self.incumbent_sol, size=n)
        for candidate in to_remove:
            self.neighbor.remove(candidate)

    def do_swap(self, n_out=1, n_in=2):
        assert isinstance(n_out, int)
        assert isinstance(n_in, int)

        self.do_n_remove(n_out)
        self.do_n_insert(n_in)  # there is possibility of removing and inserting the same member

    def gen_neighbor(self):
        # Start from incumbent solution
        self.reset_neighbor()

        # Apply some move e.g. insert, remove, swap...
        # Change this strategy accordingly
        self.do_swap()

        # Check neighbor feasibility
        self.inst.set_sol(self.neighbor)
        is_valid = self.inst.check_sol_coverage()
        return is_valid

    def search_neighbors(self, strategy: SearchStrategy = SearchStrategy.FIRST):
        assert isinstance(strategy, SearchStrategy)

        self.inst.set_sol(self.incumbent_sol)
        current_cost = self.inst.get_sol_cost()
        total_iter = 0
        total_valid = 0

        for i in range(self.max_iter_per_neighborhood):
            total_iter += 1

            is_valid = self.gen_neighbor()
            if is_valid:
                total_valid += 1

                self.neighbor, cost = self.solver.redundancy_elimination()

                self.logger.debug(f"[Update {i}]: "
                                  f"Inserted -> {[int(x) for x in self.neighbor if x not in self.incumbent_sol]}, "
                                  f"Removed -> {[int(y) for y in self.incumbent_sol if y not in self.neighbor]}, "
                                  f"Cost {current_cost} -> {cost}")
                self.run_data["valid-neighbor-cost"].append(cost)
            else:  # Avoid spending compute on unfeasible neighbors
                continue

            if cost < current_cost:
                current_cost = cost  # Update cost
                # Handle solution update
                if strategy == SearchStrategy.FIRST:  # stop after first improvement
                    self.incumbent_sol = self.neighbor.copy()
                    self.logger.info(f"Found better solution after {total_iter}!")
                    break
                else:
                    current_cost = cost
                    self.best_neighbor = self.neighbor

        if strategy == SearchStrategy.BEST:  # pick best after exhaustive neighborhood search
            self.logger.info(f"Exhausted neighborhood ({total_iter})")
            self.incumbent_sol = self.best_neighbor

        # Update instance
        self.inst.set_sol(self.incumbent_sol)
        assert self.inst.check_sol_coverage()


        # Log some stats
        self.logger.info(f"Explored {total_iter} neighbors, "
                         f"found {total_valid} feasible "
                         f"({total_valid/total_iter * 100:.2f}%)")

        # Apply post-processing
        _, cost = self.solver.redundancy_elimination()
        self.incumbent_sol = self.inst.get_sol()
        assert self.inst.check_sol_coverage()

        return current_cost

    def run(self):
        start_time = perf_counter()
        self.logger.info(f"Run with {self.strategy} strategy")

        for i in range(self.max_iter):
            cost = self.search_neighbors(strategy=self.strategy)
            self.logger.info(f"Step {i}: cost -> {cost}")

        self.improvement_step_elapsed_time = perf_counter() - start_time
        assert self.inst.check_sol_coverage()

        self.inst.set_state(State.IMPROVED)

        return self.inst.get_sol(), self.inst.get_sol_cost()

    def get_elapsed_times(self):
        elapsed_times = {
            "improvement-step": self.improvement_step_elapsed_time
        }
        return elapsed_times

    def reset_neighbor(self):
        self.neighbor = self.incumbent_sol.copy()  # shallow copy is OK since it is a list of integers

    def get_run_data(self):
        return self.run_data




if __name__== "__main__":
    dl = DataLoader()
    inst_name = "46"
    inst = dl.load_instance(inst_name)

    OUTPUT_DIR = "../output"
    if not os.path.exists(path := os.path.join(OUTPUT_DIR, "improvement-test-run")):
        os.mkdir(path)
    log_path = os.path.join(OUTPUT_DIR, "improvement-test-run", f"{inst_name}.log")

    solver = PriorityGreedySolver(instance=inst)
    solver.configure_logger(path=log_path)

    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()

    print(solver.inst.get_state())

    improver = LocalSearch(heuristic=solver, strategy=SearchStrategy.BEST)
    sol_improved, cost_improved = improver.run()

    print(improver.inst.get_state())

    print(len(sol_greedy), cost_greedy)
    print(len(p_sol_greedy), p_cost_greedy)
    print(len(sol_improved), cost_improved)
    print(improver.get_elapsed_times())

    do_plot = False
    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(improver.get_run_data()["valid-neighbor-cost"])
        plt.title("Cost - Valid neighbors")
        plt.show()
