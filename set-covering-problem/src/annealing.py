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
    # Moves / evaluation function
    PENALTY = 1.25
    SWAP_THRESHOLD = 0.5

    def __init__(self, instance: Instance,
                 max_candidates: int, max_iterations: int,
                 initial_temperature: float, final_temperature: float,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 cooling_params: dict = None,
                 equilibrium_strategy: EquilibriumStrategy = EquilibriumStrategy.STATIC,
                 logger: logging.Logger = None):

        self.logger = logger if logger else logging.getLogger()
        self.inst = instance
        assert instance.get_state() == State.SOLVED, "Initial solution must be provided"
        assert instance.check_sol_coverage(), "Provided solution is not valid"

        # unpack instance data
        self.mat = self.inst.get_mat()
        self.weights = self.inst.get_weights()
        self.n_rows = self.inst.get_rows()
        self.n_cols = self.inst.get_cols()
        self.cost_optimal = instance.get_best_known_sol_cost()

        # convert solution representation to binary vector
        sol_as_list_of_integers = self.inst.get_sol()
        self.initial_sol = np.zeros(self.n_cols, dtype=int)
        self.initial_sol[sol_as_list_of_integers] = 1
        assert self.is_complete_solution(self.initial_sol)

        # hyper-parameters
        self._max_candidates = max_candidates  # don't modify
        self.max_candidates = max_candidates  # modify for adaptive strategy
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.cooling_params = {"rate": 0.99} if cooling_params is None else cooling_params
        self.equilibrium_strategy = equilibrium_strategy
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

        # components
        self.temperature = None

        # statistics
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

    # auxiliary methods
    def do_compute_cost(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        cost = self.weights[x == 1].sum()
        return int(cost)

    def do_compute_coverage(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        selected_cols = self.mat[:, x == 1]
        covered_rows = np.bitwise_or.reduce(selected_cols, axis=-1)
        assert covered_rows.size == self.n_rows
        covered_total = covered_rows.sum()
        covered_percent = covered_total / self.n_rows
        return covered_total, covered_percent

    def is_complete_solution(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        covered_total, _ = self.do_compute_coverage(x)
        return covered_total == self.n_rows

    def do_fill_partial_solution(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        y = x.copy()
        penalty = 0
        for i in range(self.n_cols):
            if self.is_complete_solution(y):
                break
            if y[i]:
                continue
            selected_cols = self.mat[:, y == 1].copy()
            covered_rows = np.bitwise_or.reduce(selected_cols, axis=-1)
            col = self.mat[:, i]
            new_covers = col & ~covered_rows
            if new_covers.sum() > 0:
                y[i] = 1  # include if covers some previously uncovered rows
                penalty += SimulatedAnnealing.PENALTY * self.weights[i]
        return y, penalty

    def do_estimate_hyper_params(self, max_iter=100):
        current, candidate, best = self.do_initialize()
        cost_current = self.do_compute_cost(current)
        n_samples = 100

        # expected cost for first candidate -> cost of initial solution
        # this prevents the cost from exploding early on
        results = np.zeros(n_samples)
        temperatures = np.logspace(0, 2, n_samples)

        # flips
        flip_deltas = []
        pbar = tqdm(total=self.n_cols, desc="parameter estimation - flips", leave=False)
        for i in range(self.n_cols):
            x = current.copy()
            x[i] = 1 - current[i]
            if self.is_complete_solution(x):
                cost_x = self.do_compute_cost(x)
                delta = self.do_compute_delta(cost_current, cost_x)
                flip_deltas.append(delta)
            else:
                x, penalty = self.do_fill_partial_solution(x)
                cost_x = self.do_compute_cost(x)
                cost_x += penalty
                delta = self.do_compute_delta(cost_current, cost_x)
                flip_deltas.append(delta)
            pbar.update()
        pbar.close()
        # swaps
        swap_deltas = []
        ones_indices = np.where(current == 1)[0]
        zeros_indices = np.where(current == 0)[0]
        pbar = tqdm(total=len(ones_indices) * len(zeros_indices),
                    desc="parameter estimation - swaps", leave=False)
        for one_index in ones_indices:
            for zero_index in zeros_indices:
                x = current.copy()
                x[one_index] = 0
                x[zero_index] = 1
                if self.is_complete_solution(x):
                    cost_x = self.do_compute_cost(x)
                    delta = self.do_compute_delta(cost_current, cost_x)
                    swap_deltas.append(delta)
                else:
                    x, penalty = self.do_fill_partial_solution(x)
                    cost_x = self.do_compute_cost(x)
                    cost_x += penalty
                    delta = self.do_compute_delta(cost_current, cost_x)
                    swap_deltas.append(delta)
                pbar.update()
        pbar.close()
        flip_deltas = np.array(flip_deltas)
        swap_deltas = np.array(swap_deltas)

        # expectations
        for i, temp in enumerate(temperatures):
            flip_count = 0
            flip_contribution = 0
            for delta in flip_deltas:
                proba = np.exp(- delta / temp)
                flip_contribution += (delta + cost_current) * proba
                flip_count += 1
            flip_contribution /= flip_count

            swap_count = 0
            swap_contribution = 0
            for delta in swap_deltas:
                proba = np.exp(- delta / temp)
                swap_contribution += (delta + cost_current) * proba
                swap_count += 1
            swap_contribution /= swap_count

            cost_expected = (((1 - SimulatedAnnealing.SWAP_THRESHOLD) * flip_contribution)
                             + (SimulatedAnnealing.SWAP_THRESHOLD * swap_contribution))
            results[i] = cost_current - cost_expected
            print(f"temp {temp},"
                  f" flips {flip_contribution}, swaps {swap_contribution},"
                  f" diff {abs(cost_current - cost_expected)}")

        best_temp_index = (np.abs(results)).argmin()
        initial_temperature = temperatures[best_temp_index]

        proba = 0.05
        cheapest_set = self.weights.min()
        final_temperature = - cheapest_set / np.log(proba)

        if final_temperature > initial_temperature:
            final_temperature = 0.01 * initial_temperature

        rate = (final_temperature / initial_temperature)**(1/max_iter)
        return initial_temperature, final_temperature, rate


    # main methods
    def run(self):
        i = 0
        self.do_start_clock()  # statistics

        # initialization
        current, candidate, best = self.do_initialize()
        cost_current = self.do_compute_cost(current)
        cost_candidate = self.do_compute_cost(candidate)
        cost_best = self.do_compute_cost(best)

        # outer loop
        pbar = tqdm(total=self.max_iterations, desc="annealing", leave=False)
        while self.temperature > self.final_temperature and i < self.max_iterations:
            # update temperature
            self.do_cooling(i)
            j = 0
            # inner loop - search
            min_inner_cost = cost_current
            max_inner_cost = cost_current
            while not self.reached_equilibrium(j):
                candidate, penalty = self.do_next_candidate(current)
                cost_candidate = self.do_compute_cost(candidate) + penalty

                min_inner_cost = min(min_inner_cost, cost_candidate)
                max_inner_cost = max(max_inner_cost, cost_candidate)
                ratio = 1 - (np.exp( -( max_inner_cost - min_inner_cost ) / max_inner_cost ))
                max_candidates_for_next_iter = self._max_candidates + int(self._max_candidates * ratio)
                self.max_candidates = max_candidates_for_next_iter
                self.logger.info(f"max candidates {i+1} -> {self.max_candidates}")

                j += 1
                self.do_write_step()  # statistics
                self.do_read_clock()  # statistics
                self.do_write_temperature()  # statistics
                self.do_write_costs(cost_current, cost_candidate, cost_best)  # statistics
                self.do_write_sizes(current, candidate, best)  # statistics

                if self.do_accept(cost_current, cost_candidate):  # candidate either improves, or wins lottery
                    current = candidate.copy()  # update current
                    cost_current = cost_candidate
                    if cost_current < cost_best: # update best
                        best = current.copy()
                        cost_best = cost_current
                    break
            i += 1
            pbar.set_description(f"annealing -> cost {cost_current} ({cost_best} / {self.cost_optimal})")
            pbar.update()
        self.logger.info(f"reached stop condition after {i} steps "
                         f"(max steps -> {self.max_iterations})")
        assert self.is_complete_solution(best)
        return best, cost_best


    def do_next_candidate(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.n_cols
        do_swap = SimulatedAnnealing.RNG.uniform() < SimulatedAnnealing.SWAP_THRESHOLD
        candidate = x.copy()
        if do_swap:  # swap 1, 1
            ones_indices = np.where(x == 1)[0]
            zeros_indices = np.where(x == 0)[0]
            one_index = SimulatedAnnealing.RNG.choice(ones_indices)
            zero_index = SimulatedAnnealing.RNG.choice(zeros_indices)
            candidate[one_index] = 0
            candidate[zero_index] = 1
        else:  # flip
            index = SimulatedAnnealing.RNG.integers(0, self.n_cols)
            candidate[index] = 1 - candidate[index]
        if self.is_complete_solution(candidate):
            penalty = 0
            return candidate, penalty
        else:
            candidate, penalty = self.do_fill_partial_solution(candidate)
            return candidate, penalty

    def do_cooling(self, step):
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            self.temperature = self.temperature - step * self.cooling_params["rate"]
        elif self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            self.temperature = self.cooling_params["rate"] * self.temperature
        else:
            raise ValueError("Provide cooling schedule")

    def do_compute_delta(self, cost_current, cost_candidate):
        delta = cost_candidate - cost_current
        self.do_write_delta(delta)
        return delta

    def do_accept(self, cost_current, cost_candidate):
        delta = self.do_compute_delta(cost_current, cost_candidate)
        proba = np.exp(- delta / self.temperature)
        self.do_write_acceptance_proba(proba) # statistics
        self.logger.info(f"delta -> {delta:.1f} " 
                         f"acceptance proba -> {proba:.6f} "
                         f"(temperature -> {self.temperature})")
        if delta < 0:  # found improvement
            return True
        return SimulatedAnnealing.RNG.uniform() < proba

    def do_initialize(self):
        self.temperature = self.initial_temperature
        current = self.initial_sol.copy()
        candidate = self.initial_sol.copy()
        best = self.initial_sol.copy()
        return current, candidate, best

    def reached_equilibrium(self, step):
        if self.equilibrium_strategy == EquilibriumStrategy.STATIC:
            return step >= self.max_candidates
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

    def do_write_step(self):
        if len(self.stats["step"]) < 1:
            self.stats["step"].append(0)
        else:
            last_step = self.stats["step"][-1]
            self.stats["step"].append(last_step + 1)

    def do_write_delta(self, delta):
        self.stats["delta"].append(delta)

    def do_write_acceptance_proba(self, proba):
        self.stats["acceptance-proba"].append(proba)

    def do_write_costs(self, current, candidate, best):
        self.stats["current-cost"].append(current)
        self.stats["candidate-cost"].append(candidate)
        self.stats["best-cost"].append(best)

    def do_write_sizes(self, current, candidate, best):
        self.stats["current-size"].append(len(current))
        self.stats["candidate-size"].append(len(candidate))
        self.stats["best-size"].append(len(best))

    def do_write_temperature(self):
        self.stats["temperature"].append(self.temperature)

    def configure_logger(self, log_file_path):
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_cost_optimal(self):
        return self.cost_optimal

    def set_hyper_parameters(self, **kwargs):
        if initial_temperature := kwargs.get("initial_temperature", None):
            self.initial_temperature = initial_temperature
        if final_temperature := kwargs.get("final_temperature", None):
            self.final_temperature = final_temperature
        if cooling_params := kwargs.get("cooling_params", None):
            self.cooling_params = cooling_params
        if max_iterations := kwargs.get("max_iterations", None):
            self.max_iterations = max_iterations
        if max_candidates := kwargs.get("max_candidates", None):
            self.max_candidates = max_candidates
            self._max_candidates = max_candidates

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

    # don't change these... dummy values
    max_iter = 100
    max_cand = 1000
    Ti = 30
    Tf = 0.01
    alpha = (Tf / Ti)**(1 / max_iter)

    # print(f"Ti {Ti} -> Tf {Tf} (alpha = {alpha:.5f} -> {max_iter} iter)")

    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=max_cand, max_iterations=max_iter,
        initial_temperature=Ti, final_temperature=Tf,
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": alpha},
        equilibrium_strategy=EquilibriumStrategy.STATIC
    )
    sim_annealing.configure_logger(log_path)

    # configure hyper parameters
    max_iter = 200
    max_cand = 1000
    Ti, Tf, alpha = sim_annealing.do_estimate_hyper_params(max_iter)

    sim_annealing.set_hyper_parameters(
        initial_temperature=Ti,
        final_temperature=Tf,
        cooling_params={"rate": alpha},
        max_candidates=max_cand,
        max_iterations=max_iter
    )

    print(f"Ti {Ti} -> Tf {Tf} (alpha = {alpha:.5f} -> {max_iter} iter)")

    # run annealing algorithm and hope for the best
    best, cost_best = sim_annealing.run()
    cost_optimal = sim_annealing.get_cost_optimal()
    is_hit = cost_optimal == cost_best
    stats = sim_annealing.get_stats()


    import matplotlib.pyplot as plt

    time = stats["step"]

    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=2.4, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.4, ls="-", alpha=0.9)
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(cost_optimal, color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, stats["temperature"], label="temperature",
               color="tab:orange", ls="-", lw=0.8)
    # twinx.scatter(time, stats["acceptance-proba"], marker=".", color="tab:red", alpha=0.3)
    plt.ylabel("temperature")

    plt.title(f"simulated annealing \n "
              f"hit: {is_hit} -> best: {cost_best}, opt: {cost_optimal}")
    plt.show()
