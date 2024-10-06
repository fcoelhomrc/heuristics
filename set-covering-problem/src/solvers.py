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

        is_covered = coverage > 0

        total_new_covers = np.zeros(n_cols)
        for i in range(n_cols):
            if i in sol:
                new_covers = 0
            else:
                not_covered = np.bitwise_not(is_covered)
                new_covers = mat[not_covered, i].sum()
            total_new_covers[i] = new_covers
        return total_new_covers

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

    def clean_redundant(self):
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


class BetterGreedySolver(GreedySolver):

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        super().__init__(instance, logger)

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


if __name__== "__main__":
    dl = DataLoader()
    inst = dl.load_instance("42")
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

    solver = BetterGreedySolver(instance=inst)
    sol_greedy, cost_greedy = solver.greedy_heuristic()
    p_sol_greedy, p_cost_greedy = solver.clean_redundant()
    print(sol_greedy, cost_greedy)
    print(p_sol_greedy, p_cost_greedy)
