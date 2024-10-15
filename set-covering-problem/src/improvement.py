from locale import currency
from re import search

from utils import DataLoader, Instance, State
import numpy as np
import logging
from time import perf_counter
from solvers import *
from enum import Enum
from typing import Union
from tqdm import tqdm


class BestNeighbor:
    """
    Build neighborhood, exhaustively search for best neighbor
    """

    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, heuristic: Union[GreedySolver, RandomGreedySolver, PriorityGreedySolver]):

        # assert heuristic.inst.get_state() == State.SOLVED

        self.solver = heuristic
        self.logger = self.solver.logger
        self.inst = self.solver.inst # Reduce verbosity in this implementation
        self.initial_sol = self.inst.get_sol()

        self.max_iter = 3

        self.incumbent_sol = self.initial_sol.copy()
        self.neighbor = None
        self.best_neighbor = None

        self.improvement_step_elapsed_time = None
        self.run_data = {
            "feasible-neighbor-cost": []
        }

    def removal_candidates_for_swap_one(self):
        mat = self.inst.get_mat()
        ranks = []
        for j in self.incumbent_sol:
            rows_covered_by_j = mat[:, j]
            ranks.append(rows_covered_by_j.sum())
        decreasing_sort = np.asarray(ranks).argsort()[::-1]
        candidates = np.asarray(self.incumbent_sol)[decreasing_sort]
        return candidates

    def insertion_candidates_for_swap_one(self, removal_candidate):
        mat = self.inst.get_mat()
        col = mat[:, removal_candidate]
        rows_uncovered_by_removal = (col > 0)
        overlaps = np.zeros(self.inst.get_cols())
        for j in range(self.inst.get_cols()):
            if j == removal_candidate:
                overlaps[j] = 0
                continue
            rows_covered_by_j = mat[:, j]
            overlaps[j] = (rows_uncovered_by_removal & rows_covered_by_j).sum()
        increasing_sort = overlaps.argsort()
        candidates = np.asarray([i for i in range(self.inst.get_cols())])[increasing_sort]

        costs = np.asarray(self.inst.weights)[candidates]
        smaller_costs = (costs <= costs[removal_candidate])
        candidates = candidates[smaller_costs]

        return candidates

    def swap_one(self, col_in, col_out):
        sol = self.incumbent_sol.copy()
        sol.remove(col_out)
        sol.append(col_in)
        return sol

    def gen_neighbor(self, col_in, col_out):
        self.neighbor = self.swap_one(col_in, col_out)
        self.inst.set_sol(self.neighbor)
        cost = self.inst.get_sol_cost()
        is_valid = self.inst.check_sol_coverage()
        return cost, is_valid

    def search_neighbors(self):

        self.inst.set_sol(self.incumbent_sol)
        current_cost = self.inst.get_sol_cost()
        total_iter = 0
        total_feasible = 0

        to_remove = self.removal_candidates_for_swap_one()
        for col_out in to_remove:
            self.reset_neighbors()
            self.inst.set_sol(self.incumbent_sol)

            to_insert = self.insertion_candidates_for_swap_one(
                removal_candidate=col_out
            )

            for col_in in to_insert:
                total_iter += 1

                cost, is_valid = self.gen_neighbor(col_in, col_out)

                if is_valid:
                    self.run_data["feasible-neighbor-cost"].append(cost)
                    total_feasible += 1

                if is_valid and cost < current_cost:
                    self.best_neighbor = self.neighbor
                    current_cost = cost

        if self.best_neighbor is None:
            self.inst.set_sol(self.incumbent_sol)
        else:
            self.inst.set_sol(self.best_neighbor)

        assert self.inst.check_sol_coverage()

        self.logger.info(f"Explored {total_iter} neighbors, "
                         f"found {total_feasible} feasible "
                         f"({total_feasible/total_iter * 100:.2f}%)")

        _, cost = self.solver.redundancy_elimination()
        self.incumbent_sol = self.inst.get_sol()
        assert self.inst.check_sol_coverage()
        return current_cost

    def run(self):
        start_time = perf_counter()

        for i in range(self.max_iter):
            cost = self.search_neighbors()
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

    def reset_neighbors(self):
        self.neighbor = None
        self.best_neighbor = None

    def get_run_data(self):
        return self.run_data


class FirstNeighbor(BestNeighbor):

    def __init__(self, heuristic: Union[GreedySolver, RandomGreedySolver, PriorityGreedySolver]):
        super().__init__(heuristic=heuristic)

    def search_neighbors(self):

        self.inst.set_sol(self.incumbent_sol)
        current_cost = self.inst.get_sol_cost()
        total_iter = 0
        total_feasible = 0
        found_better = False

        to_remove = self.removal_candidates_for_swap_one()
        for col_out in to_remove:
            if found_better:
                break

            self.reset_neighbors()
            self.inst.set_sol(self.incumbent_sol)

            to_insert = self.insertion_candidates_for_swap_one(
                removal_candidate=col_out
            )

            for col_in in to_insert:
                total_iter += 1

                cost, is_valid = self.gen_neighbor(col_in, col_out)

                if is_valid:
                    self.run_data["feasible-neighbor-cost"].append(cost)
                    total_feasible += 1

                if is_valid and cost < current_cost:
                    self.best_neighbor = self.neighbor
                    current_cost = cost
                    found_better = True
                    break

        if self.best_neighbor is None:
            self.inst.set_sol(self.incumbent_sol)
        else:
            self.inst.set_sol(self.best_neighbor)

        assert self.inst.check_sol_coverage()

        self.logger.info(f"Explored {total_iter} neighbors, "
                         f"found {total_feasible} feasible "
                         f"({total_feasible/total_iter * 100:.2f}%)")

        _, cost = self.solver.redundancy_elimination()
        self.incumbent_sol = self.inst.get_sol()
        assert self.inst.check_sol_coverage()
        return current_cost



if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("43")

    solver = PriorityGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()

    print(solver.inst.get_state())

    # Screw up initial solution
    N = 10
    added = np.random.randint(0, solver.inst.get_cols(), size=N)
    solver.inst.get_sol().extend(list(added))

    improver = FirstNeighbor(heuristic=solver)
    sol_improved, cost_improved = improver.run()

    print(improver.inst.get_state())

    print(len(sol_greedy), cost_greedy)
    print(len(p_sol_greedy), p_cost_greedy)
    print(len(sol_improved), cost_improved)
    print(improver.get_elapsed_times())
