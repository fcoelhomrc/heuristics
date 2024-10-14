from utils import DataLoader, Instance, State
import numpy as np
import logging


class GreedySolver:

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        self.inst = instance
        self.logger = logger if logger else logging.getLogger()

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
        sol = self.inst.get_sol()
        size = len(sol)
        cost = self.inst.get_sol_cost()

        for col in sol:
            self.inst.prune_sol(col)
            is_redundant = self.inst.check_sol_coverage()
            if not is_redundant:
                self.inst.increment_sol(col)

        assert self.inst.check_sol_coverage()  # ensure pruning doesn't mess up solution

        pruned_sol = self.inst.get_sol()
        pruned_size = len(pruned_sol)
        pruned_cost = self.inst.get_sol_cost()
        self.logger.info(f"Size: {size} -> {pruned_size}")
        self.logger.info(f"Cost: {cost} -> {pruned_cost}")
        return np.asarray(pruned_sol), pruned_cost


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

        self.priority_threshold = 2

    def check_priority_usage(self):
        element_frequency = np.array(self.inst.get_total_covered_by())
        element_frequency = np.delete(element_frequency, self.inst.get_sol())
        use_priority = element_frequency.std() < self.priority_threshold
        self.logger.info(f"Priority check: {element_frequency.std()} -> "
                         f"{use_priority} (Threshold: {self.priority_threshold})")
        return use_priority

    def compute_greedy_score(self):

        greedy_score = super().compute_greedy_score()
        if self.check_priority_usage():

            sol = self.inst.get_sol()
            mat = self.inst.get_mat()
            n_cols = self.inst.get_cols()
            priority_score = np.zeros(n_cols)
            for i in range(n_cols):
                if i in sol:
                    continue
                else:
                    element_frequency = np.array(self.inst.get_total_covered_by())
                    priority_score[i] = element_frequency[mat[:, i]].sum() / element_frequency.sum()
            eps = 1e-6
            priority_score = 1 / (priority_score + eps)
            self.logger.info(f"Priority score: {priority_score}")
            return priority_score * greedy_score
        else:
            return greedy_score

if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("d1")
    print(inst.n_rows, inst.n_cols)
    print(len(inst.weights))
    print(len(inst.total_covered_by))
    print(len(inst.covered_by))
    print(inst.mat.shape)

    inst.sol = [0, 1, 2]
    print(inst.get_coverage_percent())
    print(inst.check_sol_coverage())
    print(inst.get_sol_cost())

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
