from locale import currency
from re import search

from utils import DataLoader, Instance, State
import numpy as np
import logging
from time import perf_counter
from solvers import *
from enum import Enum
from typing import Union


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

        self.max_iter = 10000
        self.moves_per_iter = 100

        self.incumbent_sol = self.initial_sol.copy()
        self.neighbor = None
        self.best_neighbor = None

        self.improvement_step_elapsed_time = None

    def random_swap(self, to_swap: list):
        sol = to_swap.copy()
        sol_in = BestNeighbor.RNG.integers(0, self.inst.get_cols(), size=1)
        sol_out = BestNeighbor.RNG.choice(sol, size=1)

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

        move = self.random_swap

        start_time = perf_counter()

        for i in range(self.max_iter):
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

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("d4")

    solver = PriorityGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()

    print(solver.inst.get_state())

    improver = BestNeighbor(heuristic=solver)
    sol_improved, cost_improved = improver.run()

    print(improver.inst.get_state())

    print(len(sol_greedy), cost_greedy)
    print(len(p_sol_greedy), p_cost_greedy)
    print(len(sol_improved), cost_improved)
    print(improver.get_elapsed_times())