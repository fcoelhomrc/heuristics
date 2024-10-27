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

class EquilibriumStrategy(Enum):
    STATIC = 0

class SimulatedAnnealing:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance,
                 max_candidates: int, max_iterations: int,
                 initial_temperature: float, final_temperature: float,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.STATIC,
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
        pbar = tqdm(total=self.max_iterations, desc="annealing", leave=False)
        while self.temperature > self.final_temperature and i < self.max_iterations:
            # update stage
            self.do_cooling(i)
            # search stage
            j = 0
            while not self.reached_equilibrium(j):
                self.do_next_candidate()
                if self.candidate is None:
                    continue
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
            pbar.set_description(f"annealing -> cost {self.sol_cost}")
            pbar.update()
        self.logger.info(f"reached stop condition after {i} steps "
                         f"(max steps -> {self.max_iterations})")


    def dry_run(self):
        self.do_initialize()
        sample_size = 1_000
        probas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        deltas = []
        pbar = tqdm(total=sample_size, desc="dry run")
        n = 0
        while n < sample_size:
            self.do_next_candidate()
            if self.candidate is None:
                continue
            n += 1
            delta = self.do_compute_delta()
            deltas.append(delta)
            pbar.update()
        delta_std = np.asarray(deltas).std()
        init_temps = (-3 / np.log(probas)) * delta_std
        print(f"dry run -> delta std {delta_std}")
        return probas, init_temps

    def evaluation_func(self, sol):
        self.inst.set_sol(sol)
        cost = self.inst.get_sol_cost()
        return cost

    def do_next_candidate(self):
        candidate = [x for x in self.sol]

        remove_options = [x for x in candidate]
        remove_options += [None]
        x_out = SimulatedAnnealing.RNG.choice(candidate, size=2, replace=True)
        insert_options = [x for x in range(self.inst.get_cols()) if x not in candidate]
        insert_options += [None]
        x_in = SimulatedAnnealing.RNG.choice(insert_options, size=2, replace=True)
        # perform swap(n_out, n_in)
        for x in x_out.tolist():
            if x is None:
                continue
            if x in candidate:
              candidate.remove(x)
            # self.logger.info(f"changes -> removed {x}")
        for x in x_in.tolist():
            if x is None:
                continue
            if x not in candidate:
                candidate.append(x)
                # self.logger.info(f"changes -> inserted {x}")

        # self.logger.info(f"cost -> current {self.evaluation_func(self.sol)}, "
        #                  f"candidate {self.evaluation_func(candidate)}")

        # perform redundancy elimination
        if self.do_check_coverage(candidate):
            candidate, _ = self.redundancy_elimination(candidate)
            self.candidate = candidate
        else:
            self.candidate = None

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

    def do_accept(self):
        self.do_write_acceptance_proba() # statistics
        if not self.do_check_coverage(self.candidate):
            return False
        delta = self.do_compute_delta()
        proba = np.exp(- delta / self.temperature)
        self.logger.info(f"delta -> {delta:.1f} " 
                         f"acceptance proba -> {proba:.6f} "
                         f"(temperature -> {self.temperature})")
        if delta < 0:
            return True
        return SimulatedAnnealing.RNG.uniform() < proba

    def do_initialize(self):
        self.temperature = self.initial_temperature
        self.sol = self.initial_sol[:]
        self.candidate = self.sol[:]
        self.best = self.sol[:]

    def do_check_coverage(self, x):
        inst = copy.deepcopy(self.inst)
        inst.set_sol(x.copy())
        return inst.check_sol_coverage()

    def redundancy_elimination(self, sol):
        inst = copy.deepcopy(self.inst)
        inst.set_sol(sol.copy())
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

    max_iter = 50
    max_cand = 1
    Ti = 30
    Tf = 1
    alpha = (Tf / Ti)**(1 / max_iter)

    print(f"Ti {Ti} -> Tf {Tf} (alpha = {alpha:.5f} -> {max_iter} iter)")

    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=max_cand, max_iterations=max_iter,
        initial_temperature=Ti, final_temperature=Tf,
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": alpha},
        equilibrium_strategy=EquilibriumStrategy.STATIC
    )
    sim_annealing.configure_logger(log_path)

    # init_probas, init_temps = sim_annealing.dry_run()
    # dry_run_res = {p: t for p,t in zip(init_probas, init_temps)}
    # print(dry_run_res)

    sim_annealing.run()

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
