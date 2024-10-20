from multiprocessing.managers import Value

from PyQt6.uic.Compiler.qobjectcreator import logger
from numpy.f2py.f90mod_rules import options

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

class CoolingSchedule(Enum):
    LINEAR = 0
    GEOMETRIC = 1

class EquilibriumStrategy(Enum):
    STATIC = 0
    ADAPTIVE = 1

class NeighborhoodStructure(Enum):
    N1 = 0
    N2 = 1
    RANDOM = 2

class SimulatedAnnealing:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance,
                 max_candidates: int, max_iterations: int,
                 initial_temperature: float, final_temperature: float,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.STATIC,
                 neighborhood_structure: NeighborhoodStructure = NeighborhoodStructure.N1,
                 logger: logging.Logger = None):

        self.inst = instance
        assert instance.get_state() == State.SOLVED, "Initial solution must be provided"
        assert instance.check_sol_coverage(), "Provided solution is not valid"

        self.logger = logger if logger else logging.getLogger()

        self.initial_sol = self.inst.get_sol()

        # Hyper-parameters
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.cooling_params = {"rate": 0.99} if cooling_params is None else cooling_params
        self.equilibrium_strategy = equilibrium_strategy
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.neighborhood_structure = neighborhood_structure

        # Components
        self.temperature = None
        self.sol = None
        self.candidate = None
        self.best = None
        self.candidate_generator = None

        # Statistics
        self.time = None
        self.stats = {
            "step": [],
            "runtime": [],
            "temperature": [],
            "delta": [],
            "current-cost": [],
            "candidate-cost": [],
            "best-cost": [],
            "current-size": [],
            "candidate-size": [],
            "best-size": [],
            "acceptance-proba": []
        }
        self.sol_cost = None
        self.candidate_cost = None
        self.best_cost = None

    def run(self):  # TODO: bug! not checking if solution is valid -> cost goes to zero
        i = 0
        self.do_start_clock()  # statistics
        self.do_initialize()
        while self.temperature > self.final_temperature:  # TODO: implement a better stopping criteria
            self.do_cooling(i)
            j = 0
            candidates = self.do_enumerate_candidates(self.neighborhood_structure)
            while not self.reached_equilibrium(j):
                self.do_next_candidate(candidates)
                if not self.do_check_if_valid_candidate():
                    logger.info(f"skipped -> candidate is not a valid solution")
                    continue

                j += 1
                self.do_write_step()  # statistics
                self.do_read_clock()  # statistics
                self.do_write_temperature()  # statistics

                if self.candidate is None:
                    self.logger.info(f"exhausted neighborhood -> {j} candidates considered")
                    break  # exhausted candidates
                self.do_compute_costs()  # statistics
                self.do_write_costs()  # statistics
                self.do_write_sizes()  # statistics
                if self.do_accept():  # candidate either improves, or wins lottery
                    self.do_update_sol()
                    self.do_update_best()
                    break
            i += 1
        logger.info(f"reached stop condition after {i} steps "
                    f"(max steps -> {self.max_iterations})")

    def evaluation_func(self, sol):
        self.inst.set_sol(sol)
        cost = self.inst.get_sol_cost()
        return cost

    def do_insert(self, insert):
        if insert is not None:
            self.candidate.append(insert)

    def do_remove(self, remove):
        if remove in self.candidate and remove is not None:
            self.candidate.remove(remove)

    def do_swap(self, remove, insert):
        for i in remove:
            self.do_remove(i)
        for i in insert:
            self.do_insert(i)

    def do_enumerate_candidates(self, neighborhood_structure: NeighborhoodStructure):
        sol = self.sol[:]  # starting point for generating neighborhood
        options_to_insert = [i for i in range(self.inst.get_cols())] + [None]  # all insertion options

        if neighborhood_structure == NeighborhoodStructure.N1:
            to_remove = sol[:]
            coverage_percent_if_removed = []
            to_remove_cost = []
            for i in to_remove:
                self.inst.set_sol(sol.copy())
                self.inst.prune_sol(i)
                coverage_percent_if_removed.append(self.inst.get_coverage_percent())
                to_remove_cost.append(self.inst.get_sol_cost())
            sort_by_coverage = np.argsort(coverage_percent_if_removed)
            to_remove = list(np.asarray(to_remove)[sort_by_coverage][::-1])
            coverage_percent_if_removed = list(np.asarray(coverage_percent_if_removed)[sort_by_coverage][::-1])
            to_remove_cost = list(np.asarray(to_remove_cost)[sort_by_coverage][::-1])

            swaps = []
            for i, p, c in zip(to_remove, coverage_percent_if_removed, to_remove_cost):
                for j in options_to_insert:
                    if j is None:  # adds  option of not inserting
                        swaps.append((i, j))
                        continue

                    if j in sol:
                        continue

                    self.inst.set_sol(sol.copy())
                    self.inst.prune_sol(i)
                    self.inst.increment_sol(j)
                    if self.inst.get_coverage_percent() < p:
                        continue
                    if self.inst.get_sol_cost() > c:
                        continue
                    swaps.append((i, j))

            self.logger.info(f"generated N1 with size {len(swaps)} "
                             f"(max. candidates -> {self.max_candidates}) "
                             f"(max. % explored -> {100 * min(1., self.max_candidates/len(swaps)):.5f} %)")

            for swap in swaps:
                yield swap

        elif neighborhood_structure == NeighborhoodStructure.N2:
            raise NotImplementedError  # TODO: implement N2 structure (should contain N1)

        elif neighborhood_structure == NeighborhoodStructure.RANDOM:
            to_remove = SimulatedAnnealing.RNG.choice(
                sol, size=(self.max_candidates, 2)
            )

            options_to_insert_without_sol = [i for i in options_to_insert if i not in sol]
            to_insert = SimulatedAnnealing.RNG.choice(
                options_to_insert_without_sol, size=(self.max_candidates, 2)
            )

            swaps = []
            for i in range(self.max_candidates):
                remove_1 = to_remove[i, 0]
                remove_2 = None # to_remove[i, 1]
                insert_1 = to_insert[i, 0]
                insert_2 = None # to_insert[i, 1]
                swap = (
                    (
                        int(remove_1) if remove_1 else None,
                        int(remove_2) if remove_2 else None
                    ),
                    (
                        int(insert_1) if insert_1 else None,
                        int(insert_2) if insert_2 else None,
                    )
                )
                swaps.append(swap)

            for swap in swaps:
                yield swap

        else:
            raise ValueError("Provide neighborhood structure")

    def do_next_candidate(self, generator):
        try:
            swap = next(generator)
            insert = swap[0]
            remove = swap[1]
            self.candidate = self.sol[:]
            self.do_swap(insert, remove)
            return True
        except StopIteration:
            return False

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
        self.do_write_delta(delta)
        return delta

    def do_check_if_valid_candidate(self):
        self.inst.set_sol(self.candidate.copy())
        return self.inst.check_sol_coverage()

    def do_accept(self):

        self.do_write_acceptance_proba()

        delta = self.do_compute_delta()
        if delta <= 0:
            logger.info(f"accepted -> candidate improves current solution")
            return True
        else:
            self.logger.info(f"acceptance proba -> {np.exp(- delta / self.temperature):.6f} "
                             f"(temperature -> {self.temperature})")
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

    def get_initial_temperature_estimate(self, acceptance_proba=0.99):
        deltas = self.stats["delta"]
        if len(deltas) < 1:
            return
        sigma = np.asarray(deltas).std()
        k = -3 / np.log(acceptance_proba)
        return k * sigma

    # Results
    def get_best(self):
        return self.best.copy()

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

    def do_write_delta(self, delta):
        self.stats["delta"].append(delta)

    def do_write_acceptance_proba(self):
        if not self.do_check_if_valid_candidate():
            proba = None
        delta = self.do_compute_delta()
        if delta <= 0:
            proba = None
        else:
            proba = np.exp(- delta / self.temperature)
        self.stats["acceptance-proba"].append(proba)

    def do_write_costs(self):
        self.stats["current-cost"].append(self.sol_cost)
        self.stats["candidate-cost"].append(self.candidate_cost)
        self.stats["best-cost"].append(self.best_cost)

    def do_write_sizes(self):
        self.stats["current-size"].append(len(self.sol))
        self.stats["candidate-size"].append(len(self.candidate))
        self.stats["best-size"].append(len(self.best))

    def do_write_temperature(self):
        self.stats["temperature"].append(self.temperature)

    def do_compute_costs(self):
        self.inst.set_sol(self.sol)
        self.sol_cost = self.inst.get_sol_cost()

        self.inst.set_sol(self.candidate)
        self.candidate_cost = self.inst.get_sol_cost()

        self.inst.set_sol(self.best)
        self.best_cost = self.inst.get_sol_cost()

    def configure_logger(self, log_file_path):
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)


if __name__ == "__main__":
    dl = DataLoader()
    inst_name = "46"
    inst = dl.load_instance(inst_name)

    OUTPUT_DIR = "../output"
    if not os.path.exists(path := os.path.join(OUTPUT_DIR, "annealing-test-run")):
        os.mkdir(path)
    log_path = os.path.join(OUTPUT_DIR, "annealing-test-run", f"{inst_name}.log")

    solver = GreedySolver(instance=inst)
    solver.configure_logger(path=log_path)
    sol_greedy, cost_greedy = solver.greedy_heuristic()

    # reset instance
    inst = dl.load_instance(inst_name)
    inst.set_sol(list(sol_greedy))
    inst.set_state(State.SOLVED)

    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=10, max_iterations=1_000,
        initial_temperature=60.0, final_temperature=1,
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": 0.99},
        neighborhood_structure=NeighborhoodStructure.RANDOM
    )
    sim_annealing.configure_logger(log_path)

    sim_annealing.run()

    suggested_initial_temperature = sim_annealing.get_initial_temperature_estimate(0.99)

    best = sim_annealing.get_best()
    stats = sim_annealing.get_stats()


    import matplotlib.pyplot as plt

    time = stats["step"]

    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=1.5, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.2, ls="-")
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=1.0, ls="-")
    plt.axhline(inst.get_best_known_sol_cost(), color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, np.asarray(stats["temperature"]) / sim_annealing.initial_temperature, label="temperature",
               color="tab:orange", ls="-", lw=0.8)
    # twinx.scatter(time, stats["acceptance-proba"], marker=".", color="tab:red", alpha=0.3)
    plt.ylabel("temperature")

    plt.title(f"simulated annealing \n "
              f"suggested initial temperature -> {suggested_initial_temperature:.5f}")
    plt.show()

    ####

    plt.plot(time, stats["best-size"], label="best-size",
             color="tab:green", lw=1.5, ls="-")
    plt.plot(time, stats["current-size"], label="current-size",
             color="k", lw=1.2, ls="-")
    plt.plot(time, stats["candidate-size"], label="candidate-size",
             color="tab:blue", lw=1.0, ls="-")

    plt.xlabel("step")
    plt.ylabel("sizes")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, np.asarray(stats["temperature"]) / sim_annealing.initial_temperature, label="temperature",
               color="tab:orange", ls="-", lw=0.8)
    twinx.scatter(time, stats["acceptance-proba"], marker=".", color="tab:red", alpha=0.3)
    plt.ylabel("temperature")

    plt.title(f"simulated annealing \n "
              f"suggested initial temperature -> {suggested_initial_temperature:.5f}")
    # plt.show()


