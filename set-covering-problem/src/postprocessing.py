from utils import Instance, State
import numpy as np
import logging

class RedundancyElimination:

    def __init__(self, instance: Instance, logger: logging.Logger = None):
        assert instance.get_state() == State.SOLVED

        self.inst = instance
        self.init_sol = self.inst.get_sol()
        self.init_cost = self.inst.get_sol_cost()
        self.init_size = len(self.inst.get_sol())

        self.mat = self.inst.get_mat()
        self.weights = self.inst.get_weights()

        self.logger = logger if logger else logging.getLogger()

        self.candidates = []
        self.ranks = []
        self.removed = []

    def do_elimination(self):
        cost = True
        while cost:
            cost = self.do_elimination_step()
        self.logger.info(f"Removed {len(self.removed)} redundancies")
        self.logger.info(f"Cost {self.init_cost} -> {self.inst.get_sol_cost()}")
        self.logger.info(f"Size {self.init_size} -> {len(self.inst.get_sol())}")

    def do_elimination_step(self):
        """
        Finds redundancy with the largest cost x coverage overlap, and removes it
        :return: Cost after elimination. None if no elimination occurred.
        """
        self.find_redundancies()

        if len(self.candidates) < 1:
            return None

        self.rank_candidates()
        assert len(self.ranks) == len(self.candidates)

        self.eliminate_top_candidate()
        assert self.inst.check_sol_coverage()

        self.do_reset()
        return self.inst.get_sol_cost()

    def eliminate_top_candidate(self):
        sort_mask = np.asarray(self.ranks).argsort()
        sorted_candidates = np.asarray(self.candidates)[sort_mask]
        top_candidate = sorted_candidates[-1]

        self.inst.prune_sol(top_candidate)
        self.removed.append(top_candidate)  # for tracking

    def rank_candidates(self):
        size = len(self.candidates)
        maxi = max(self.candidates)
        for col in self.candidates:
            candidate_elements = self.mat[:, col].reshape(-1, 1)
            # other_candidates_elements = np.delete(self.mat, col, axis=1)
            overlap = (candidate_elements & self.mat)[:, self.candidates].sum()
            cost = self.weights[col]
            self.ranks.append(overlap * cost)

    def find_redundancies(self):
        for col in self.init_sol:
            self.inst.prune_sol(col)
            is_redundant = self.inst.check_sol_coverage()
            if is_redundant:
                self.candidates.append(col)
            self.inst.increment_sol(col)

    def do_reset(self):
        self.candidates.clear()
        self.ranks.clear()