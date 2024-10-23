from utils import State, Instance, DataLoader
from improvement import LocalSearch, SearchStrategy

import logging
import os
import copy
from tqdm import tqdm
from time import perf_counter
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
        self.seeds = Grasp.RNG.integers(0, 1000, n_solutions)  # seeds in parallel programs are tricky

        # hyper parameters
        assert isinstance(n_solutions, int)
        assert isinstance(alpha, float)

        self.n_solutions = n_solutions
        self.alpha = min(alpha, 1)  # % best scores to sample from

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
                self.logger.info(f"process {idx} has completed -> size: {len(result)}")
                pbar.update()

        self.do_drop_repeated_solutions()
        assert self.do_validate_solutions()

    def do_construct_solution(self, idx):
        inst = copy.deepcopy(self.inst)

        rng = np.random.default_rng(seed=self.seeds[idx])
        starter_size = 1  # promote diversity among initial solutions
        starter = rng.integers(0, inst.get_cols(), size=starter_size).tolist()
        for x in starter:
            inst.increment_sol(x)
        while not inst.check_sol_coverage():
            scores = self.do_compute_greedy_scores(inst)
            x = self.do_choose_next(scores, seed=self.seeds[idx])
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
        cheapest = None
        for i in inst.get_sol():
            cost = weights[i]
            if cheapest is None:
                cheapest = cost
                continue
            if cost < cheapest:
                cheapest = cost
                continue
        if cheapest is None:
            return total_new_covers / np.asarray(weights)
        return (total_new_covers * cheapest) / np.asarray(weights)

    def do_choose_next(self, scores: np.ndarray, seed: int):
        rng = np.random.default_rng(seed=seed)  # ensure seed is not fixed, to get diverse solutions
        top_alpha = int(self.alpha * len(scores))
        ranks = np.argsort(scores)[::-1]  # decreasing score
        top_alpha_ranks = ranks[:top_alpha]
        choice = int(rng.choice(top_alpha_ranks, size=1)[0])
        return choice

    def do_validate_solutions(self):
        for solution in self.solutions:
            self.inst.set_sol(solution)
            if not self.inst.check_sol_coverage():
                return False
        return True

    def do_drop_repeated_solutions(self):
        unique_solutions = set(tuple(sorted(sublist)) for sublist in self.solutions)
        self.solutions = [list(item) for item in unique_solutions]
        self.logger.warning(f"generated {self.n_solutions} solutions "
                            f"-> {len(self.solutions)} are unique")

    # user interface / run statistics
    def get_solutions(self):
       return copy.deepcopy(self.solutions)

    def get_solutions_costs(self):
        costs = []
        for x in self.solutions:
            self.inst.set_sol(x)
            costs.append(self.inst.get_sol_cost())
        return np.array(costs)

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

        self.max_runtime = 1 * 60  # seconds
        self.max_iter = 200
        self.optimized_solutions = []

    def run(self):
        super().run()  # generate initial solutions
        pbar = tqdm(total=len(self.solutions), desc="vnd", leave=False)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.do_local_search, self.solutions[i], i
                ): i for i in range(len(self.solutions))
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result = future.result()
                assert isinstance(result[0], list)
                self.optimized_solutions.append(result[0])
                self.logger.info(f"process {idx} has completed -> "
                                 f"cost: {result[1]} ({self.inst.get_best_known_sol_cost()}) "
                                 f"size: {len(result[0])}")
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
                        self.logger.info(f"process {idx} -> found improvement in n2!")
                    elapsed = perf_counter() - start  # ensure that runtime is contained
                    if elapsed > self.max_runtime:
                        self.logger.warning(f"process {idx} -> search in n2 timed out!")
                        break
            except StopIteration:
                pass
            if found_improvement:
                n_iter += 1
            else:  # exhausted all options, halt search
                self.logger.info(f"process {idx} -> could not find improvements, halting")
                break
        inst = copy.deepcopy(self.inst)
        inst.set_sol(current)
        cost = inst.get_sol_cost()
        return current, cost

    def do_generate_n1(self, solution: list):
        inst = copy.deepcopy(self.inst)
        mat = inst.get_mat()
        weights = inst.get_weights()
        rng = np.random.default_rng()  # ensure seed is not fixed, to get diverse solutions

        # remove
        percent_covered_if_removed = np.zeros(len(solution))
        for i, x in enumerate(solution):
            s = solution.copy()
            s.remove(x)
            inst.set_sol(s)
            percent_covered_if_removed[i] = inst.get_coverage_percent()
        sort_by_decreasing_percent_covered = percent_covered_if_removed.argsort()[::-1]
        remove_options = np.asarray(solution)[sort_by_decreasing_percent_covered]
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
                if weights[x_in] < weights[x] and overlap > 0:
                    yield [x, x_in]
                    continue

    def do_generate_n2(self, solution: list):
        inst = copy.deepcopy(self.inst)
        mat = inst.get_mat()
        weights = inst.get_weights()
        rng = np.random.default_rng()  # ensure seed is not fixed, to get diverse solutions

        # remove
        remove_options = []
        percent_covered_if_removed = []
        for i, x in enumerate(solution):
            for j, y in enumerate(solution + [None]):  # adds option to remove one or two
                if j < i and y is not None:
                    continue  # removing (1, 2) is equivalent to (2, 1)
                s = copy.deepcopy(solution)
                s.remove(x)
                if x != y and y is not None:
                    s.remove(y)
                inst.set_sol(s)
                remove_options.append([x, y])
                percent_covered_if_removed.append(inst.get_coverage_percent())
        sort_by_decreasing_percent_covered = np.asarray(
            percent_covered_if_removed
        ).argsort()[::-1]
        remove_options = np.asarray(remove_options)[sort_by_decreasing_percent_covered]
        remove_options = remove_options.tolist()

        # insert
        for x, y in remove_options:
            if x is None:
                elements_removed = mat[:, y]
                cost_removed = weights[y]
            elif y is None:
                elements_removed = mat[:, x]
                cost_removed = weights[x]
            else:
                elements_removed = (mat[:, x] | mat[:, y])
                cost_removed = weights[x] + weights[y]

            insert_options = [i for i in range(inst.get_cols())]
            insert_options = [None] + [x for _, x in sorted(zip(weights, insert_options))]
            for i, x_in in enumerate(insert_options):
                if x_in is not None:
                    if weights[x_in] > cost_removed:
                        continue  # save time
                for j, y_in in enumerate(insert_options):
                    if (j < i) and ((x_in is not None) or (y_in is not None)):
                        continue  # insert (1, 2) equivalent to (2, 1)
                    if {x, y} == {x_in, y_in}:
                        continue  # skip identity move (remove and insert the same members)
                    if (x_in is None) and (y_in is not None):
                        if weights[y_in] > cost_removed:
                            continue  # save time
                    if (x_in is None) or (y_in is None):
                        yield [x, y, x_in, y_in]  #
                        continue
                    cost_inserted = weights[x_in] + weights[y_in]
                    elements_inserted = (mat[:, x_in] | mat[:, y_in])
                    overlap = (elements_removed & elements_inserted).sum()
                    if cost_inserted < cost_removed and overlap > 0:
                        yield [x, y, x_in, y_in]
                        continue

    @staticmethod
    def do_swap(solution: list, remove: list, insert: list):
        x = copy.deepcopy(solution)
        if len(remove) == 2 and remove[0] == remove[1]:
            x.remove(remove[0])
        else:
            for i in remove:
                if i is not None:
                    x.remove(i)
        for i in insert:
            if i is not None:
                x.append(i)
        return x

    def is_candidate_better(self, current: list, candidate: list):
        inst = copy.deepcopy(self.inst)
        inst.set_sol(current)
        current_cost = inst.get_sol_cost()
        inst.set_sol(candidate)
        candidate_cost = inst.get_sol_cost()
        if not inst.check_sol_coverage():
            return False
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

    # # tune hyper parameters
    # SIZE = inst.get_cols()
    # N_ALPHAS = 4
    # ALPHAS = np.linspace(1/SIZE, 25/SIZE, N_ALPHAS)
    # print(ALPHAS)
    # N_SOLUTIONS = 3
    # COSTS = np.zeros((N_ALPHAS, 3))  # min, mean, max
    # for i in range(N_ALPHAS):
    #     grasp = Grasp(instance=inst,
    #                   alpha=float(ALPHAS[i]), n_solutions=N_SOLUTIONS)
    #     grasp.run()
    #     print(grasp.alpha, grasp.alpha * SIZE)
    #
    #     costs = grasp.get_solutions_costs()
    #     costs = np.array(costs)
    #     COSTS[i, 0] = costs.min()
    #     COSTS[i, 1] = costs.mean()
    #     COSTS[i, 2] = costs.max()
    #
    #     print(f"COSTS: {len(costs), costs.mean(), costs.max()}")
    #
    # import matplotlib.pyplot as plt
    # plt.plot(ALPHAS * SIZE, COSTS[:, 1], c="k", lw=2.0, ls="-")
    # plt.plot(ALPHAS * SIZE, COSTS[:, 0], c="tab:blue", lw=2.0, ls="-")
    # plt.plot(ALPHAS * SIZE, COSTS[:, 2], c="tab:blue", lw=2.0, ls="-")
    # plt.fill_between(ALPHAS * SIZE, COSTS[:, 0], COSTS[:, 2],
    #                  color="tab:blue", alpha=0.6)
    # # plt.axhline(inst.get_best_known_sol_cost(), c="k", lw=2.0, ls="--")
    # plt.xlabel("top-K sampled")
    # plt.ylabel("cost")
    # plt.show()

    K = 1
    alpha = K / inst.get_cols()
    print(f"alpha -> {alpha}")
    n_solutions = 16
    grasp = GraspVND(instance=inst,
                     alpha=alpha, n_solutions=n_solutions)

    grasp.configure_logger(log_path)

    max_iter = 100
    grasp.max_iter = max_iter
    max_time = 3 * 60  # around 1h for 42 instances
    grasp.max_runtime = max_time
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