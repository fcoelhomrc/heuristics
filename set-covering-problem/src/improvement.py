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

        assert heuristic.inst.get_state() == State.SOLVED

        self.solver = heuristic
        self.logger = self.solver.logger
        self.inst = self.solver.inst # Reduce verbosity in this implementation
        self.initial_sol = self.inst.get_sol()

        self.max_iter = 100
        self.moves_per_iter = 1000

        self.incumbent_sol = self.initial_sol.copy()
        self.neighbor = None
        self.best_neighbor = None

        self.improvement_step_elapsed_time = None
        self.run_data = {
            "feasible-neighbor-cost": []
        }

    def random_insert(self, to_insert: list):
        sol = to_insert.copy()
        sol_in = BestNeighbor.RNG.integers(0, self.inst.get_cols(), size=1)

        sol.append(sol_in[0])
        return sol

    def random_double_insert(self, to_insert: list):
        sol = to_insert.copy()
        sol_in = BestNeighbor.RNG.integers(0, self.inst.get_cols(), size=2)

        sol.extend(list(sol_in))
        return sol

    def random_swap(self, to_swap: list):
        sol = to_swap.copy()
        sol_in = BestNeighbor.RNG.integers(0, self.inst.get_cols(), size=1)
        sol_out = BestNeighbor.RNG.choice(sol, size=1)

        sol.remove(sol_out[0])
        sol.append(sol_in[0])
        return sol

    def smart_swap(self, to_swap: list):
        mat = self.inst.get_mat()
        weights = self.inst.get_weights()
        candidates = to_swap.copy()
        candidates_ranks = []

        for col in candidates:
            candidate_elements = mat[:, col].reshape(-1, 1)
            other_candidates_elements = np.delete(mat, col, axis=1)
            overlap = (candidate_elements & other_candidates_elements)[:, candidates].sum()
            cost = weights[col]
            candidates_ranks.append(overlap * cost)

        # Sort candidates by rank
        candidates = np.array(candidates)
        candidates_ranks = np.array(candidates_ranks)

        sort_by_rank = np.argsort(candidates_ranks)
        candidates = candidates[sort_by_rank]

        # Convert into probability distribution
        candidates_ranks /= candidates_ranks.sum()

        # Sample candidate to be removed based on ranks
        sol_out = BestNeighbor.RNG.choice(candidates, size=1, p=candidates_ranks)

        # Sample candidate to be added based on cost
        weights = np.array(weights)
        weights_normalized = weights / weights.sum()
        sol_in = BestNeighbor.RNG.choice([j for j in range(self.inst.get_cols())],
                                         size=1, p=weights_normalized)

        sol = to_swap.copy()
        sol.remove(sol_out[0])
        sol.append(sol_in[0])
        return sol

    def gen_neighbor(self, move):
        self.neighbor = move(self.incumbent_sol)
        self.inst.set_sol(self.neighbor)
        cost = self.inst.get_sol_cost()
        is_valid = self.inst.check_sol_coverage()
        return cost, is_valid

    def search_neighbors(self, move):

        self.inst.set_sol(self.incumbent_sol)
        current_cost = self.inst.get_sol_cost()

        for i in range(self.moves_per_iter):
            self.reset_neighbors()
            self.inst.set_sol(self.incumbent_sol)

            cost, is_valid = self.gen_neighbor(move)
            if is_valid:
                self.run_data["feasible-neighbor-cost"].append(cost)
            if is_valid and cost < current_cost:
                self.best_neighbor = self.neighbor
                current_cost = cost

        if self.best_neighbor is None:
            self.inst.set_sol(self.incumbent_sol)
        else:
            self.inst.set_sol(self.best_neighbor)

        assert self.inst.check_sol_coverage()

        _, cost = self.solver.redundancy_elimination()
        self.incumbent_sol = self.inst.get_sol()
        assert self.inst.check_sol_coverage()
        return current_cost

    def run(self):

        move = self.random_double_insert

        start_time = perf_counter()

        for i in tqdm(range(self.max_iter)):
            cost = self.search_neighbors(move=move)
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

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("c3")

    solver = PriorityGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()

    print(solver.inst.get_state())

    # Screw up initial solution
    # N = 100
    # added = np.random.randint(0, solver.inst.get_cols(), size=N)
    # solver.inst.get_sol().extend(list(added))

    improver = BestNeighbor(heuristic=solver)
    sol_improved, cost_improved = improver.run()

    print(improver.inst.get_state())

    print(len(sol_greedy), cost_greedy)
    print(len(p_sol_greedy), p_cost_greedy)
    print(len(sol_improved), cost_improved)
    print(improver.get_elapsed_times())
