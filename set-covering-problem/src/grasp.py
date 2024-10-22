from time import perf_counter

from utils import State, Instance, DataLoader
from improvement import LocalSearch, SearchStrategy

import logging
import os
import copy
from tqdm import tqdm
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Grasp:
    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance,
                 alpha: float = 0.80, n_solutions: int = 5,
                 logger: logging.Logger = None):

        self.inst = instance
        self.logger = logger if logger else logging.getLogger()

        # hyper parameters
        assert isinstance(n_solutions, int)
        assert isinstance(alpha, float) and 0. < alpha < 1.0

        self.n_solutions = n_solutions
        self.alpha = alpha  # % best scores to sample from

        # components
        self.solutions = []

    def run(self):
        pbar = tqdm(total=self.n_solutions, desc="grasp", leave=False)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.do_construct_solution, i
                ): i for i in range(self.n_solutions)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result = future.result()
                assert isinstance(result, list)
                self.solutions.append(result)
                self.logger.info(f"process {idx} has completed -> {result}")
                pbar.update()
        assert self.do_validate_solutions()

    def do_construct_solution(self, i=0):
        inst = copy.deepcopy(self.inst)
        while not inst.check_sol_coverage():
            scores = self.do_compute_greedy_scores(inst)
            x = self.do_choose_next(scores)
            inst.increment_sol(x)
        solution = inst.get_sol().copy()
        return solution

    @staticmethod
    def do_compute_greedy_scores(inst: Instance):
        coverage = inst.get_coverage_per_element()
        sol = inst.get_sol()
        mat = inst.get_mat()
        n_cols = inst.get_cols()
        weights = inst.get_weights()
        is_covered = coverage > 0
        total_new_covers = np.zeros(n_cols)
        for i in range(n_cols):
            if i in sol:
                new_covers = 0
            else:
                not_covered = np.bitwise_not(is_covered)
                new_covers = mat[not_covered, i].sum()
            total_new_covers[i] = new_covers
        return total_new_covers / np.asarray(weights)

    def do_choose_next(self, scores: np.ndarray):
        rng = np.random.default_rng()  # ensure seed is not fixed, to get diverse solutions
        top_alpha = int(self.alpha * len(scores))
        ranks = scores.argsort()[::-1]  # decreasing score
        return int(rng.choice(ranks[:top_alpha], size=1)[0])

    def do_validate_solutions(self):
        for solution in self.solutions:
            self.inst.set_sol(solution)
            if not self.inst.check_sol_coverage():
                return False
        return True

    # user interface / run statistics
    def get_solutions(self):
       return copy.deepcopy(self.solutions)

    def get_best(self):
        inst = copy.deepcopy(self.inst)
        best = None
        best_cost = None
        for solution in self.solutions:
            inst.set_sol(solution)
            cost = inst.get_sol_cost()
            if best_cost is None:
                best = solution
                best_cost = cost
            if cost < best_cost:
                best = solution
                best_cost = cost
        return best, best_cost

    def configure_logger(self, log_file_path):
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)



class GraspVND(Grasp):

    def __init__(self, instance: Instance,
                 alpha: float = 0.80, n_solutions: int = 5,
                 logger: logging.Logger = None):
        super().__init__(instance=instance, alpha=alpha, n_solutions=n_solutions)

        self.max_runtime = 10 * 60  # seconds
        self.max_iter = 100
        self.optimized_solutions = []

    def run(self):
        super().run()  # generate initial solutions
        pbar = tqdm(total=self.n_solutions, desc="vnd", leave=False)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.do_local_search, self.solutions[i], i
                ): i for i in range(self.n_solutions)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result = future.result()
                assert isinstance(result, list)
                self.optimized_solutions.append(result)
                self.logger.info(f"process {idx} has completed -> {result}")
                pbar.update()
        assert self.do_validate_optimized_solutions()


    def do_local_search(self, solution, idx):
        start = perf_counter()
        elapsed = 0
        n_iter = 0
        current = solution.copy()
        while n_iter < self.max_iter and elapsed < self.max_runtime:
            n1 = self.do_generate_n1(current)  # search for improvement in small neighborhood
            found_improvement = False
            try:
                while not found_improvement:
                    swaps = next(n1)
                    candidate = self.do_swap(
                        current, remove=[swaps[0]], insert=[swaps[1]]
                    )
                    if self.is_candidate_better(current, candidate):
                        current = candidate.copy()
                        found_improvement = True
                    elapsed = perf_counter() - start  # ensure that runtime is contained
                    if elapsed > self.max_runtime:
                        break
            except StopIteration:
                pass
            if found_improvement:
                n_iter += 1
                continue  # re-center search on new solution
            n2 = self.do_generate_n2(current)  # ...or keep looking in larger neighborhood
            self.logger.info(f"process {idx} -> searching n2 at iter {n_iter}")
            try:
                while not found_improvement:
                    swaps = next(n2)
                    candidate = self.do_swap(
                        current, remove=swaps[:2], insert=swaps[2:]
                    )
                    if self.is_candidate_better(current, candidate):
                        current = candidate.copy()
                        found_improvement = True
                    elapsed = perf_counter() - start  # ensure that runtime is contained
                    if elapsed > self.max_runtime:
                        break
            except StopIteration:
                pass
            if found_improvement:
                n_iter += 1
            else:  # exhausted all options, halt search
                break
        return current

    def do_generate_n1(self, solution: list):
        inst = copy.deepcopy(self.inst)
        mat = inst.get_mat()
        weights = inst.get_weights()
        rng = np.random.default_rng()  # ensure seed is not fixed, to get diverse solutions

        # remove
        percent_uncovered_if_removed = np.zeros(len(solution))
        for i, x in enumerate(solution):
            s = solution.copy()
            s.remove(x)
            inst.set_sol(s)
            percent_uncovered_if_removed[i] = inst.get_coverage_percent()
        sort_by_decreasing_percent_uncovered = percent_uncovered_if_removed.argsort()[::-1]
        remove_options = np.asarray(solution)[sort_by_decreasing_percent_uncovered]
        remove_options = remove_options.tolist()

        # insert
        for x in remove_options:
            insert_options = [i for i in range(inst.get_cols())] + [None]
            rng.shuffle(insert_options)
            for x_in in insert_options:
                if x_in == x:
                    continue
                if x_in is None:
                    yield [x, x_in]  # remove but don't insert back
                    continue
                overlap = (mat[:, x] & mat[:, x_in]).sum()
                if weights[x_in] <= weights[x] and overlap > 0:
                    yield [x, x_in]
                    continue

    def do_generate_n2(self, solution: list):
        inst = copy.deepcopy(self.inst)
        mat = inst.get_mat()
        weights = inst.get_weights()
        rng = np.random.default_rng()  # ensure seed is not fixed, to get diverse solutions

        # remove
        remove_options = []
        percent_uncovered_if_removed = []
        for i, x in enumerate(solution):
            for j, y in enumerate(solution):
                if j < i:
                    continue  # removing (1, 2) is equivalent to (2, 1)
                s = copy.deepcopy(solution)
                s.remove(x)
                if x != y:
                    s.remove(y)
                inst.set_sol(s)
                remove_options.append([x, y])
                percent_uncovered_if_removed.append(inst.get_coverage_percent())
        sort_by_decreasing_percent_uncovered = np.asarray(
            percent_uncovered_if_removed
        ).argsort()[::-1]
        remove_options = np.asarray(remove_options)[sort_by_decreasing_percent_uncovered]
        remove_options = remove_options.tolist()

        # insert
        for x, y in remove_options:
            insert_options = [i for i in range(inst.get_cols())] + [None]
            rng.shuffle(insert_options)
            for x_in in insert_options:
                for y_in in insert_options:
                    if {x, y} == {x_in, y_in}:  # sets
                        continue
                    if x_in is None or y_in is None:
                        yield [x, y, x_in, y_in]
                        continue
                    elements_removed = (mat[:, x] | mat[:, y])
                    elements_inserted = (mat[:, x_in] | mat[:, y_in])
                    overlap = (elements_removed & elements_inserted).sum()

                    cost_removed = weights[x] + weights[y]
                    cost_inserted = weights[x_in] + weights[y_in]
                    if cost_inserted <= cost_removed and overlap > 0:
                        yield [x, y, x_in, y_in]
                        continue

    @staticmethod
    def do_swap(solution: list, remove: list, insert: list):
        x = copy.deepcopy(solution)
        if len(remove) == 2 and remove[0] == remove[1]:
            x.remove(remove[0])
        else:
            for i in remove:
                x.remove(i)
        for i in insert:
            if i is not None:
                x.append(i)
        return x

    def is_candidate_better(self, current: list, candidate: list):
        inst = copy.deepcopy(self.inst)
        inst.set_sol(candidate)
        if not inst.check_sol_coverage():
            return False
        candidate_cost = inst.get_sol_cost()
        inst.set_sol(current)
        current_cost = inst.get_sol_cost()
        return candidate_cost < current_cost

    def do_validate_optimized_solutions(self):
        for solution in self.optimized_solutions:
            self.inst.set_sol(solution)
            if not self.inst.check_sol_coverage():
                return False
        return True

    def get_optimized_solutions(self):
        return copy.deepcopy(self.optimized_solutions)

    def get_optimized_best(self):
        inst = copy.deepcopy(self.inst)
        best = None
        best_cost = None
        for solution in self.optimized_solutions:
            inst.set_sol(solution)
            cost = inst.get_sol_cost()
            if best_cost is None:
                best = solution
                best_cost = cost
            if cost < best_cost:
                best = solution
                best_cost = cost
        return best, best_cost


if __name__ == "__main__":
    dl = DataLoader()
    inst_name = "42"
    inst = dl.load_instance(inst_name)

    OUTPUT_DIR = "../output"
    if not os.path.exists(path := os.path.join(OUTPUT_DIR, "grasp-test-run")):
        os.mkdir(path)
    log_path = os.path.join(OUTPUT_DIR, "grasp-test-run", f"{inst_name}.log")

    alpha = 0.10
    n_solutions = 50
    grasp = GraspVND(instance=inst,
                     alpha=alpha, n_solutions=n_solutions)

    grasp.configure_logger(log_path)

    grasp.run()

    for solution in grasp.get_solutions():
        print(solution)

    for solution in grasp.get_optimized_solutions():
        print(solution)

    init_best, init_cost = grasp.get_best()
    best, best_cost = grasp.get_optimized_best()
    print(f"init cost -> {init_cost}\n"
          f"best cost -> {best_cost}\n"
          f"optimal cost -> {inst.get_best_known_sol_cost()}")