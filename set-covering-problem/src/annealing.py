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
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.STATIC):

        self.inst = instance
        assert instance.get_state() == State.SOLVED, "Initial solution must be provided"
        assert instance.check_sol_coverage(), "Provided solution is not valid"

        self.initial_sol = self.inst.get_sol()
        self.temperature = None  # TODO: initialize temperature

        # Hyper-parameters
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.cooling_params = {"rate": 0.99} if cooling_params is None else cooling_params
        self.equilibrium_strategy = equilibrium_strategy

        self.sol = None
        self.candidate = None
        self.best_candidate = None

        # Statistics
        self.time = None
        self.stats = {
            "step": [],
            "runtime": [],
            "temperature": [],
            "best-cost": [],
            "current-cost": [],
            "candidate-cost": []
        }

    def run(self):
        i = 0
        self.do_start_clock()
        while True:  # TODO: implement stopping criteria (end cooling)
            self.do_cooling(i)
            j = 0
            while not self.reached_equilibrium(j):   # TODO: implement stopping criteria (equilibrium condition)
                j += 1
                self.do_write_step(j)  # statistics
                self.do_read_clock()  # statistics
                self.do_write_temperature()  # statistics
                self.do_generate_candidate()
                self.do_write_costs()  # statistics
                if self.do_accept():
                    self.do_update_sol()
                    break
            i += 1

    def evaluation_func(self, sol):
        return  # TODO: implement evaluation function

    def do_generate_candidate(self):
        return  # TODO: apply some move (modify self.candidate directly)

    def do_cooling(self, step):
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            self.temperature = self.temperature - step * self.cooling_params["rate"]
        elif self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            self.temperature = self.cooling_params["rate"] * self.temperature

        return  # TODO: implement cooling schedule (modify self.temperature directly)

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

    def do_update_sol(self):
        self.sol = self.candidate[:]  # Prevents shallow copy

    def reached_equilibrium(self, step):
        if step >= self.max_candidates:
            return True
        return

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
        return  # TODO: write all costs to stats (best, current, candidate)

    def do_write_temperature(self):
        self.stats["temperature"].append(self.temperature)
