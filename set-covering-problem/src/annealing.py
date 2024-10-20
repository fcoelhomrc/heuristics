from multiprocessing.managers import Value

from utils import State, Instance, DataLoader
from solvers import GreedySolver, RandomGreedySolver, PriorityGreedySolver
from postprocessing import RedundancyElimination
from improvement import SearchStrategy, LocalSearch

import os
from typing import Union
import logging
import numpy as np
from tqdm import tqdm
from enum import Enum
from time import perf_counter

class CoolingSchedule(Enum):
    LINEAR = 0
    GEOMETRIC = 1

class EquilibriumStrategy(Enum):
    STATIC = 0
    ADAPTIVE = 1

class SimulatedAnnealing:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance, logger: logging.Logger,
                 max_candidates: int, max_iterations: int,
                 initial_temperature: int,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.STATIC):

        self.inst = instance
        assert instance.get_state() == State.SOLVED, "Initial solution must be provided"
        assert instance.check_sol_coverage(), "Provided solution is not valid"

        self.logger = logger

        self.initial_sol = self.inst.get_sol()

        # Hyper-parameters
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.cooling_params = {"rate": 0.99} if cooling_params is None else cooling_params
        self.equilibrium_strategy = equilibrium_strategy
        self.initial_temperature = initial_temperature

        # Components
        self.temperature = None
        self.sol = None
        self.candidate = None
        self.best = None

        # Statistics
        self.time = None
        self.stats = {
            "step": [],
            "runtime": [],
            "temperature": [],
            "current-cost": [],
            "candidate-cost": [],
            "best-cost": [],
        }
        self.sol_cost = None
        self.candidate_cost = None
        self.best_cost = None

    def run(self):
        i = 0
        self.do_start_clock()  # statistics
        self.do_initialize()
        while True:  # TODO: implement stopping criteria (end cooling)
            self.do_cooling(i)
            j = 0
            while not self.reached_equilibrium(j):
                j += 1
                self.do_write_step(j)  # statistics
                self.do_read_clock()  # statistics
                self.do_write_temperature()  # statistics
                self.do_generate_candidate()
                self.do_compute_costs()  # statistics
                self.do_write_costs()  # statistics
                if self.do_accept():  # candidate either improves, or wins lottery
                    self.do_update_sol()
                    self.do_update_best()
                    break
            i += 1

    def evaluation_func(self, sol):
        return  # TODO: implement evaluation function

    def do_n_insert(self, n=2):
        cols = int(self.inst.get_cols())
        to_insert =  LocalSearch.RNG.choice(
            [col for col in range(cols) if col not in self.candidate], size=n
        )
        for i in to_insert:
            self.candidate.append(i)

    def do_n_remove(self, n=1):
        sol_ext = self.sol + [-1]  # adds chance of not removing a member
        to_remove =  SimulatedAnnealing.RNG.choice(sol_ext, size=n)
        for i in to_remove:
            if i >= 0:
                self.candidate.remove(i)

    def do_swap(self, n_remove=1, n_insert=2):
        self.do_n_remove(n_remove)
        self.do_n_insert(n_insert)

    def do_generate_candidate(self):
        self.candidate = self.sol[:]
        self.do_swap()

    def do_cooling(self, step):
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            self.temperature = self.temperature - step * self.cooling_params["rate"]
        elif self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            self.temperature = self.cooling_params["rate"] * self.temperature
        else:
            raise ValueError("Provide cooling schedule")

    def do_compute_delta(self):
        candidate_fitness = self.evaluation_func(self.candidate)
        sol_fitness = self.evaluation_func(self.sol)
        delta = candidate_fitness - sol_fitness
        return delta

    def do_accept(self):
        delta = self.do_compute_delta()
        if delta <= 0:
            return True
        else:
            return SimulatedAnnealing.RNG.uniform() < np.exp(- delta / self.temperature)

    def do_initialize(self):
        self.temperature = self.initial_temperature
        self.sol = self.initial_sol[:]
        self.best = self.sol[:]

    def do_update_sol(self):
        self.sol = self.candidate[:]  # Prevents shallow copy

    def do_update_best(self):
        if self.sol_cost < self.best_cost:
            self.best = self.sol[:]

    def reached_equilibrium(self, step):
        if self.equilibrium_strategy == EquilibriumStrategy.STATIC:
            return step >= self.max_candidates
        elif self.equilibrium_strategy == EquilibriumStrategy.ADAPTIVE:
            raise NotImplementedError  #TODO: implement adaptive equilibrium
        else:
            raise ValueError("Provide an equilibrium strategy")

    def get_stats(self):
        return self.stats

    # Statistics
    def do_start_clock(self):
        self.time = perf_counter()

    def do_read_clock(self):
        now = perf_counter()
        self.stats["runtime"].append(now - self.time)
        self.time = now

    def do_write_step(self, step):
        self.stats["step"].append(step)

    def do_write_costs(self):
        self.stats["current-cost"].append(self.sol_cost)
        self.stats["candidate-cost"].append(self.candidate_cost)
        self.stats["best-cost"].append(self.best_cost)

    def do_write_temperature(self):
        self.stats["temperature"].append(self.temperature)

    def do_compute_costs(self):
        self.inst.set_sol(self.sol)
        self.sol_cost = self.inst.get_sol_cost()

        self.inst.set_sol(self.candidate)
        self.candidate_cost = self.inst.get_sol_cost()

        self.inst.set_sol(self.best)
        self.best_cost = self.inst.get_sol_cost()

