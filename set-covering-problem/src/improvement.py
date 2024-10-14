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

        self.max_iter = 10
        self.moves_per_iter = 1000

        self.incumbent_sol = self.initial_sol.copy()
        self.incumbent_sol_cost = self.get_cost(self.incumbent_sol)

        self.neighborhood = []

        self.best_neighbor = None
        self.best_neighbor_cost = None

        self.improvement_heuristic_elapsed_time = None

    def update(self):
        return self.inst.get_sol()

    def get_cost(self, sol):
        self.inst.set_sol(sol)
        return self.inst.get_sol_cost()

    def insert_move(self, col):
        self.inst.increment_sol(col)

    def remove_move(self, col):
        self.inst.prune_sol(col)

    def swap_move(self, col_in, col_out):
        self.inst.increment_sol(col_in[0])
        self.inst.prune_sol(col_out[0])

    def sample_from_solution(self, n=1):
        return BestNeighbor.RNG.choice(self.inst.get_sol(), size=n)

    def sample_from_all(self, n=1):
        return BestNeighbor.RNG.integers(0, self.inst.get_rows(), size=n)

    def build_neighborhood(self):
        for i in range(self.moves_per_iter):

            self.swap_move(col_in=self.sample_from_all(),
                           col_out=self.sample_from_solution())
            self.neighborhood.append(self.update())

    def explore_neighborhood(self):
        neighborhood_size = len(self.neighborhood)
        assert neighborhood_size > 0
        costs = []
        for i in range(neighborhood_size):
            neighbor = self.neighborhood[i]
            self.inst.set_sol(neighbor)

            if self.inst.check_sol_coverage():
                costs.append([i, self.get_cost(neighbor)])

        if len(costs) > 0:
            costs = np.array(costs)
            min_cost_index = costs[:, 1].argmin()

            self.best_neighbor = self.neighborhood[costs[costs[min_cost_index, 0]]]
            self.best_neighbor_cost = costs[min_cost_index, 1]
        else:
            self.best_neighbor = None
            self.best_neighbor_cost = None

    def run(self):
        self.logger.info(f"Running improvement heuristic for {self.max_iter} steps")
        start_time = perf_counter()
        for i in range(self.max_iter):
            self.build_neighborhood()
            self.explore_neighborhood()

            print(f"{self.incumbent_sol_cost} -> {self.best_neighbor_cost}")

            self.logger.info(f"Step {i} - Cost: {self.incumbent_sol_cost}"
                             f" -> Best Neighbor: {self.best_neighbor_cost}")
            if self.best_neighbor is None or self.best_neighbor_cost is None:
                continue
            if self.best_neighbor_cost < self.incumbent_sol_cost:
                self.incumbent_sol = self.best_neighbor
                self.incumbent_sol_cost = self.best_neighbor_cost

                self.inst.set_sol(self.incumbent_sol)
               # _ = self.solver.redundancy_elimination()  # Perform CE between each step
        self.improvement_heuristic_elapsed_time = start_time - perf_counter()

        self.inst.set_sol(self.incumbent_sol)

        assert self.inst.check_sol_coverage()
        self.inst.set_state(State.IMPROVED)

        return np.asarray(self.inst.get_sol()), self.inst.get_sol_cost()

    def get_elapsed_times(self):
        elapsed_times = {
            "improvement-heuristic": self.improvement_heuristic_elapsed_time
        }
        return elapsed_times

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("42")

    solver = PriorityGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()

    print(solver.inst.get_state())

    improver = BestNeighbor(heuristic=solver)
    sol_improved, cost_improved = improver.run()

    print(improver.inst.get_state())

    print(sol_greedy, cost_greedy)
    print(p_sol_greedy, p_cost_greedy)
    print(sol_improved, cost_improved)
    print(improver.get_elapsed_times())