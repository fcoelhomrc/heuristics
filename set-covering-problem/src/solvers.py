from utils import DataLoader, Instance, State
from postprocessing import RedundancyElimination
import numpy as np
import logging
from time import perf_counter
import os


class GreedySolver:

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        self.inst = instance
        self.logger = logger if logger else logging.getLogger()

        self.top_k_redundancy_elimination = 5

        self.constructive_step_elapsed_time = None
        self.redundancy_elimination_elapsed_time = None

    def compute_greedy_score(self):
        coverage = self.inst.get_coverage_per_element()
        sol = self.inst.get_sol()
        mat = self.inst.get_mat()
        n_cols = self.inst.get_cols()
        weights = self.inst.get_weights()

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

    def greedy_step(self):
        """
        Add element which covers most uncovered sets
        :return:
        """
        score = self.compute_greedy_score()
        return score.argmax()

    def greedy_heuristic(self):
        total_iter = 0
        iter_coverage = None
        iter_cost = None
        start_time = perf_counter()

        while not self.inst.check_sol_coverage():
            best_candidate = self.greedy_step()
            self.inst.increment_sol(best_candidate)
            # Stats / progress
            total_iter += 1
            iter_coverage = self.inst.get_coverage_percent()
            iter_cost = self.inst.get_sol_cost()
            self.logger.info(f"Step {total_iter}: "
                             f"size: {len(self.inst.get_sol())}, "
                             f"coverage: {100 * iter_coverage:.2f}%, "
                             f"cost: {iter_cost}")

        self.inst.set_state(State.SOLVED)
        self.constructive_step_elapsed_time = perf_counter() - start_time
        return np.asarray(self.inst.get_sol()), self.inst.get_sol_cost()

    def configure_logger(self, path):
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def redundancy_elimination(self):
        start_time = perf_counter()
        engine = RedundancyElimination(instance=self.inst, logger=self.logger)
        engine.do_elimination()
        self.redundancy_elimination_elapsed_time = perf_counter() - start_time

        assert self.inst.check_sol_coverage()

        return [int(x) for x in self.inst.get_sol()], self.inst.get_sol_cost()

    @staticmethod
    def _rank_candidates_for_redundancy_elimination(candidates, candidates_ranks, mat, weights):
        # Rank candidates
        # More overlap, more likely to be redundant
        # More cost, more attractive for removal
        for col in candidates:
            candidate_elements = mat[:, col].reshape(-1, 1)
            other_candidates_elements = np.delete(mat, col, axis=1)
            overlap = (candidate_elements & other_candidates_elements)[:, candidates].sum()
            cost = weights[col]
            candidates_ranks.append(overlap * cost)

    def _prune_solution_for_redundancy_elimination(self, candidates, sol):
        # Prune critical members of solution
        for col in sol.copy():
            self.inst.prune_sol(col)
            is_redundant = self.inst.check_sol_coverage()
            if is_redundant:
                candidates.append(col)
            self.inst.increment_sol(col)

    def get_elapsed_times(self):
        elapsed_times = {
            "constructive-step": self.constructive_step_elapsed_time,
            "redundancy-elimination": self.redundancy_elimination_elapsed_time
        }
        return elapsed_times

class RandomGreedySolver(GreedySolver):

    SEED = 42
    RNG = np.random.default_rng(seed=SEED)

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        super().__init__(instance, logger)
        self.prob = 0.1  # prob of picking among top candidates
        self.top_threshold = 0.8  # %best score to be considered top candidate
        self.total_random_steps = 0

    def greedy_step(self):
        score = self.compute_greedy_score()

        if RandomGreedySolver.RNG.uniform() < self.prob:
            threshold = self.top_threshold * score
            top_candidates = np.nonzero(score > threshold)
            self.total_random_steps += 1
            return RandomGreedySolver.RNG.choice(top_candidates)[0]
        else:
            return score.argmax()

    def get_total_random_steps(self):
        if self.inst.get_state() == State.SOLVED:
            self.logger.info(f"Solved solution used {self.total_random_steps} random steps")
        return self.total_random_steps

    def greedy_heuristic(self):
        sol, cost = super().greedy_heuristic()
        _ = self.get_total_random_steps()
        return sol, cost


class PriorityGreedySolver(GreedySolver):

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        super().__init__(instance, logger)

        self.priority_threshold = 5

    def check_priority_usage(self):
        mat = self.inst.get_mat()
        candidate_cols = np.delete(mat, self.inst.get_sol(), axis=1)
        element_frequency = candidate_cols.sum(axis=1)
        use_priority = element_frequency.std() < self.priority_threshold
        # self.logger.info(f"Priority check: {element_frequency.std()} -> "
        #                  f"{use_priority} (Threshold: {self.priority_threshold})")
        return use_priority

    def compute_greedy_score(self):
        greedy_score = super().compute_greedy_score()
        if self.check_priority_usage():
            mat = self.inst.get_mat()
            n_cols = self.inst.get_cols()
            priority_score = np.zeros(n_cols)
            for i in range(n_cols):
                element_frequency = np.array(self.inst.get_total_covered_by())
                priority_score[i] = element_frequency[mat[:, i]].sum() / element_frequency.sum()
            priority_score = - np.log(priority_score)
            # self.logger.info(f"Priority score: {priority_score}")
            return priority_score * greedy_score
        else:
            return greedy_score

if __name__== "__main__":
    dl = DataLoader()
    inst_name = "42"
    inst = dl.load_instance(inst_name)

    OUTPUT_DIR = "../output"
    if not os.path.exists(path := os.path.join(OUTPUT_DIR, "improvement-test-run")):
        os.mkdir(path)
    log_path = os.path.join(OUTPUT_DIR, "improvement-test-run", f"{inst_name}.log")

    # solver = GreedySolver(instance=inst)
    # sol_greedy, cost_greedy = solver.greedy_heuristic()
    # print(sol_greedy, cost_greedy)

    # solver = BetterGreedySolver(instance=inst)
    # sol_greedy, cost_greedy = solver.greedy_heuristic()
    # p_sol_greedy, p_cost_greedy = solver.clean_redundant()
    # print(sol_greedy, cost_greedy)
    # print(p_sol_greedy, p_cost_greedy)

    solver = PriorityGreedySolver(instance=inst)
    solver.configure_logger(path=log_path)

    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()
    print(sol_greedy, cost_greedy)
    print(p_sol_greedy, p_cost_greedy)
