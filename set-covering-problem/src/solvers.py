from utils import DataLoader, Instance, State
import numpy as np
import logging
from time import perf_counter


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
        self.logger.setLevel(logging.DEBUG)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def redundancy_elimination(self):
        if not self.inst.get_state() == State.SOLVED:
            return

        start_time = perf_counter()

        mat = self.inst.get_mat()
        weights = self.inst.get_weights()
        sol = self.inst.get_sol()
        candidates = []

        self._prune_solution_for_redundancy_elimination(candidates, sol)

        candidates_ranks = []

        self._rank_candidates_for_redundancy_elimination(candidates, candidates_ranks, mat, weights)

        # Sort candidates by rank
        candidates = np.array(candidates)
        candidates_ranks = np.array(candidates_ranks)

        sort_by_rank = np.argsort(candidates_ranks)
        candidates = candidates[sort_by_rank]

        top_k = min(len(candidates), self.top_k_redundancy_elimination)
        top_k_candidates = candidates[-top_k:]

        top_k_elements_to_prune = []
        top_k_elements_to_prune_cost = []

        for candidate in top_k_candidates:
            assert self.inst.check_sol_coverage()

            to_prune = [candidate]
            to_prune_cost = weights[candidate]
            has_redundant_cols = True

            while has_redundant_cols:

                for col in to_prune:
                    self.inst.prune_sol(col)

                current_cost = 0
                current_col = None
                for col in self.inst.get_sol():
                    self.inst.prune_sol(col)

                    if not self.inst.check_sol_coverage():
                        self.inst.increment_sol(col)
                        continue

                    cost = weights[col]

                    if current_cost < cost:
                        current_cost = cost
                        current_col = col

                    self.inst.increment_sol(col)

                for col in to_prune:
                    self.inst.increment_sol(col)

                if current_col is None:
                    has_redundant_cols = False
                else:
                    to_prune.append(current_col)
                    to_prune_cost += current_cost

            top_k_elements_to_prune.append(to_prune)
            top_k_elements_to_prune_cost.append(to_prune_cost)

        for elements_to_prune, cost in zip(top_k_elements_to_prune, top_k_elements_to_prune_cost):
            self.logger.info(f"Redundancy elimination: {elements_to_prune} -> cost: {cost}")

        best_elements_to_prune = top_k_elements_to_prune[np.argmax(np.asarray(top_k_elements_to_prune_cost))]

        size_before_pruning = len(self.inst.get_sol())
        cost_before_pruning = self.inst.get_sol_cost()
        for col in best_elements_to_prune:
            self.inst.prune_sol(col)
        size_after_pruning = len(self.inst.get_sol())
        cost_after_pruning = self.inst.get_sol_cost()
        self.logger.info(f"Size: {size_before_pruning} -> {size_after_pruning}")
        self.logger.info(f"Cost: {cost_before_pruning} -> {cost_after_pruning}")

        assert self.inst.check_sol_coverage()

        self.redundancy_elimination_elapsed_time = perf_counter() - start_time
        return np.asarray(self.inst.get_sol()), self.inst.get_sol_cost()

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
        self.logger.info(f"Priority check: {element_frequency.std()} -> "
                         f"{use_priority} (Threshold: {self.priority_threshold})")
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
            self.logger.info(f"Priority score: {priority_score}")
            return priority_score * greedy_score
        else:
            return greedy_score

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("42")

    # solver = GreedySolver(instance=inst)
    # sol_greedy, cost_greedy = solver.greedy_heuristic()
    # print(sol_greedy, cost_greedy)

    # solver = BetterGreedySolver(instance=inst)
    # sol_greedy, cost_greedy = solver.greedy_heuristic()
    # p_sol_greedy, p_cost_greedy = solver.clean_redundant()
    # print(sol_greedy, cost_greedy)
    # print(p_sol_greedy, p_cost_greedy)

    solver = PriorityGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.redundancy_elimination()
    print(sol_greedy, cost_greedy)
    print(p_sol_greedy, p_cost_greedy)
