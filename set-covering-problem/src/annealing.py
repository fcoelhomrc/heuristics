import copy

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
    NON_MONOTONIC = 2

class EquilibriumStrategy(Enum):
    STATIC = 0
    ADAPTIVE = 1
    EXHAUSTIVE = 2

class MoveType(Enum):
    INSERT = 0
    REMOVE = 1


class SimulatedAnnealing:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance,
                 max_candidates: int, max_iterations: int,
                 initial_temperature: float, final_temperature: float,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.EXHAUSTIVE,
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

        # Components
        self.temperature = None
        self.sol = None
        self.candidate = None
        self.best = None
        self.move_type = None
        self.candidate_options = None

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
            "acceptance-proba": [],
        }
        self.sol_cost = None
        self.candidate_cost = None
        self.best_cost = None

    def run(self):
        i = 0
        self.do_start_clock()  # statistics
        self.do_initialize()
        while self.temperature > self.final_temperature and i < self.max_iterations:  # TODO: implement a better stopping criteria
            # update stage
            self.do_cooling(i)
            self.do_reset_candidate_pool()
            # search stage
            j = 0
            while not self.reached_equilibrium(j):
                self.do_next_candidate()
                if self.candidate is None:
                    break

                j += 1
                self.do_write_step()  # statistics
                self.do_read_clock()  # statistics
                self.do_write_temperature()  # statistics
                self.do_compute_costs()  # statistics
                self.do_write_costs()  # statistics
                self.do_write_sizes()  # statistics

                if self.do_accept():  # candidate either improves, or wins lottery
                    self.do_update_sol()
                    self.do_update_best()
                    break
            i += 1
        self.logger.info(f"reached stop condition after {i} steps "
                         f"(max steps -> {self.max_iterations})")

    def dry_run(self, max_iterations: int):
        n_test_pts = 50
        test_temperatures = np.logspace(2, -2, n_test_pts)
        test_scores = np.zeros(n_test_pts)
        data_for_final_temperature = np.zeros((n_test_pts, 3))  # cost, log proba, delta
        for i in range(n_test_pts):
            n_iter = 0
            x = 0
            self.do_initialize()
            self.do_reset_candidate_pool()
            while self.candidate is not None:
                self.do_next_candidate()
                if self.candidate is None:
                    break
                delta = self.do_compute_delta()
                proba = np.exp(-delta / test_temperatures[i])
                self.inst.set_sol(self.candidate)
                cost = self.inst.get_sol_cost()
                data_for_final_temperature[i, 0] = cost
                data_for_final_temperature[i, 1] = np.log(proba)
                data_for_final_temperature[i, 2] = delta
                if self.move_type == MoveType.REMOVE:
                    x += (1 - proba) * cost
                elif self.move_type == MoveType.INSERT:
                    self.inst.set_sol(self.candidate)
                    x += proba * cost
                else:
                    raise ValueError(f"Invalid move {self.move_type}")
                n_iter += 1
            x /= n_iter
            test_scores[i] = x
        initial_temperature = test_temperatures[np.abs(test_scores).argmin()]
        final_temperature = 0.01 * initial_temperature
        assert final_temperature < initial_temperature

        if self.cooling_schedule == CoolingSchedule.LINEAR:
            rate = (final_temperature - initial_temperature) / max_iterations
        elif self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            rate = np.exp( (np.log(final_temperature) - np.log(initial_temperature)) / max_iterations )
        else:
            raise ValueError("Provide cooling schedule")

        # reset statistics
        for _, stat in self.stats.items():
            stat.clear()

        return initial_temperature, final_temperature, {"rate": rate}

    def evaluation_func(self, sol):
        self.inst.set_sol(sol)
        cost = self.inst.get_sol_cost()
        return cost

    def do_reset_candidate_pool(self):
        self.candidate_options = [i for i in range(self.inst.get_cols())] + [None]

    def do_next_candidate(self):
        sol = self.sol[:]
        while True:
            if len(self.candidate_options) < 1:
                self.candidate = None  # exhausted options, move on
                break

            candidate = sol.copy()
            sample = SimulatedAnnealing.RNG.choice(
                self.candidate_options, size=1
            )
            sample = int(sample[0])
            self.candidate_options.remove(sample)

            if sample in sol:
                candidate.remove(sample)
                self.move_type = MoveType.REMOVE
            else:
                candidate.append(sample)
                self.move_type = MoveType.INSERT

            if self.do_check_if_valid_operation(candidate):
                # apply redundancy elimination
                candidate, _ = self.redundancy_elimination()
                self.candidate = candidate
                break

    def do_cooling(self, step):
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            self.temperature = self.temperature - step * self.cooling_params["rate"]
            self.logger.info(f"temperature updated -> {self.temperature:.3f} / target: {self.final_temperature}")
        elif self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            self.temperature = self.cooling_params["rate"] * self.temperature
        elif self.cooling_schedule == CoolingSchedule.NON_MONOTONIC:
            is_below_threshold = self.temperature < self.cooling_params["refuel_threshold"] * self.initial_temperature
            if is_below_threshold and not self.cooling_params["active"]:
                self.cooling_params["active"] = True
                if SimulatedAnnealing.RNG.uniform() < self.cooling_params["refuel_proba"]:
                    self.temperature = self.cooling_params["refuel_rate"] * self.temperature
                else:
                    self.temperature = self.cooling_params["rate"] * self.temperature
            else:
                self.temperature = self.cooling_params["rate"] * self.temperature
        else:
            raise ValueError("Provide cooling schedule")

    def do_compute_delta(self):
        candidate_fitness = self.evaluation_func(self.candidate)
        sol_fitness = self.evaluation_func(self.sol)
        delta = candidate_fitness - sol_fitness
        self.do_write_delta(delta)
        return delta

    def do_check_if_valid_operation(self, sol):
        self.inst.set_sol(sol.copy())
        return self.inst.check_sol_coverage()

    def do_accept(self):
        self.do_write_acceptance_proba() # statistics
        delta = self.do_compute_delta()
        if delta <= 0:
            return True
        proba = min(1., np.exp(- delta / self.temperature))
        if self.move_type == MoveType.INSERT:
            self.logger.info(f"{self.move_type} "
                             f"acceptance proba -> {proba:.6f} "
                             f"(temperature -> {self.temperature})")
            return SimulatedAnnealing.RNG.uniform() < proba
        elif self.move_type == MoveType.REMOVE:
            self.logger.info(f"{self.move_type} "
                             f"acceptance proba -> {(1 - proba):.6f} "
                             f"(temperature -> {self.temperature})")
            return SimulatedAnnealing.RNG.uniform() < (1 - proba)
        else:
            raise ValueError(f"Invalid move {self.move_type}")

    def do_initialize(self):
        self.temperature = self.initial_temperature
        self.sol = self.initial_sol[:]
        self.candidate = self.sol[:]  # TODO: this might introduce a bug
        self.best = self.sol[:]

    def redundancy_elimination(self):
        inst = copy.deepcopy(self.inst)
        inst.set_sol(self.sol)
        engine = RedundancyElimination(instance=inst, logger=self.logger)
        engine.do_elimination()
        assert inst.check_sol_coverage()
        return inst.get_sol(), inst.get_sol_cost()

    def do_update_sol(self):
        self.sol = self.candidate[:]  # prevents shallow copy

    def do_update_best(self):
        self.do_compute_costs()  # make sure costs are up to date
        if self.sol_cost < self.best_cost:
            self.best = self.sol[:]

    def reached_equilibrium(self, step):
        if self.equilibrium_strategy == EquilibriumStrategy.STATIC:
            return step >= self.max_candidates
        elif self.equilibrium_strategy == EquilibriumStrategy.ADAPTIVE:
            raise NotImplementedError  #TODO: implement adaptive equilibrium
        elif self.equilibrium_strategy == EquilibriumStrategy.EXHAUSTIVE:
            return self.candidate is None
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
        delta = self.do_compute_delta()
        proba = min(1., np.exp(- delta / self.temperature))
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

    def set_hyper_parameters(self, **kwargs):
        if initial_temperature := kwargs.get("initial_temperature", None):
            self.initial_temperature = initial_temperature
        if final_temperature := kwargs.get("final_temperature", None):
            self.final_temperature = final_temperature
        if cooling_params := kwargs.get("cooling_params", None):
            self.cooling_params = cooling_params

if __name__ == "__main__":
    dl = DataLoader()
    inst_name = "42"
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

    max_iter = 5_000
    max_cand = 10 * inst.get_cols()
    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=max_cand, max_iterations=max_iter,
        initial_temperature=100.0, final_temperature=0.001,
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": 0.99}
    )
    sim_annealing.configure_logger(log_path)

    # # estimate hyper parameters
    # init_temp, final_temp, cooling = sim_annealing.dry_run(
    #     max_iterations=max_iter
    # )

    # # test new cooling schedule
    # sim_annealing.cooling_schedule = CoolingSchedule.NON_MONOTONIC
    # cooling["refuel_threshold"] = 0.75
    # cooling["refuel_rate"] = 1.05
    # cooling["refuel_proba"] = 0.20
    # cooling["active"] = False
    #
    # sim_annealing.set_hyper_parameters(
    #     initial_temperature=init_temp,
    #     final_temperature=final_temp,
    #     cooling_params=cooling
    # )

    # solve instance
    sim_annealing.run()

    # sim_annealing.logger.info(f"dry run estimates -> "
    #                           f"init temp: {init_temp:.2f} "
    #                           f"final temp: {final_temp:.2f} "
    #                           f"rate: {cooling["rate"]:.5f}")
    best = sim_annealing.get_best()
    stats = sim_annealing.get_stats()

    sim_annealing.logger.info(f"init size {len(sim_annealing.initial_sol)} "
                              f"-> curr size {len(sim_annealing.sol)}")

    inst.set_sol(best)
    assert inst.check_sol_coverage()
    best_cost, optimal_cost = inst.get_sol_cost(), inst.get_best_known_sol_cost()
    is_hit = best_cost == optimal_cost

    import matplotlib.pyplot as plt

    time = stats["step"]

    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=1.5, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.2, ls="-")
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(inst.get_best_known_sol_cost(), color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, stats["temperature"], label="temperature",
               color="tab:orange", ls="-", lw=0.8)
    # twinx.scatter(time, stats["acceptance-proba"], marker=".", color="tab:red", alpha=0.3)
    plt.ylabel("temperature")

    plt.title(f"simulated annealing \n "
              f"hit: {is_hit} -> best: {best_cost}, opt: {optimal_cost}")
    plt.show()

    ###
    # proba = np.asarray(stats["acceptance-proba"])
    # proba = proba[(1e-3 < proba) & (proba < 0.999)]
    # plt.plot(proba, label="insert",
    #          color="tab:orange", lw=1.5, ls="-")
    # plt.plot(1. - proba, label="remove",
    #          color="tab:blue", lw=1.5, ls="-")
    #
    # plt.xlabel("step")
    # plt.ylabel("proba")
    # plt.legend()
    #
    # plt.title(f"simulated annealing \n "
    #           f"hit: {is_hit} -> best: {best_cost}, opt: {optimal_cost}")
    # plt.show()
