import pandas as pd

from vnsearch import VariableNeighborhoodSearch
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

import warnings
warnings.filterwarnings('ignore')




results = dict(
    {
        "instance": [],
        "runtime": [],
        "k": [],
        "init-size": [],
        "best-size": [],
        "init-cost": [],
        "best-cost": [],
        "optimal-cost": [],
        "hit": [],
        "relative-error": [],
    }
)

dl = DataLoader()
available_inst = dl.list_instances()
results["instance"].extend(available_inst)


OUTPUT_DIR = "../output"
if not os.path.exists(path := os.path.join(OUTPUT_DIR, "vnsearch-run")):
    os.mkdir(path)

calibration_instances = {
    "42": None, "51": None, "61": None, "a1": None, "b1": None, "c1": None, "d1": None,
}

pbar = tqdm(total=len(available_inst),
            desc="instance", position=0, leave=False)

for inst_name in available_inst:
    start = perf_counter()

    log_path = os.path.join(OUTPUT_DIR, "vnsearch-run", f"{inst_name}.log")
    dl = DataLoader()
    inst = dl.load_instance(inst_name)
    solver = GreedySolver(instance=inst)
    solver.configure_logger(path=log_path)
    sol_greedy, cost_greedy = solver.greedy_heuristic()

    # reset instance
    inst = dl.load_instance(inst_name)
    inst.set_sol(list(sol_greedy))
    inst.set_state(State.SOLVED)

    # run the VNS
    N = inst.get_cols()
    frac = 1
    max_iter = 100
    max_iter_ls = {
        1: int(frac * N),
        2: int(frac * N),
        3: int(frac * N),
    }
    # print(f"max_iter_ls: {max_iter_ls}")
    vns = VariableNeighborhoodSearch(instance=inst,
                                     max_iter_local_search=max_iter_ls, max_iterations=max_iter
                                     )
    vns.configure_logger(log_path)

    best, best_cost = vns.run()
    optimal_cost = vns.get_cost_optimal()
    hit = optimal_cost == best_cost
    stats = vns.get_stats()


    results["runtime"].append(perf_counter() - start)

    init_size = len(vns.initial_sol)
    best_size = len(best)

    init_cost = vns.do_compute_cost(vns.initial_sol)
    relative_error = abs(optimal_cost - best_cost) / optimal_cost

    results["init-size"].append(init_size)

    results["k"].append(stats["k"])
    results["best-size"].append(best_size)
    results["init-cost"].append(init_cost)
    results["best-cost"].append(best_cost)
    results["optimal-cost"].append(optimal_cost)
    results["hit"].append(hit)
    results["relative-error"].append(relative_error)

    assert vns.is_complete_solution(best)  # sanity check

    save_fig_path = os.path.join(OUTPUT_DIR, "vnsearch-run", f"{inst_name}.png")

    import matplotlib.pyplot as plt

    time = stats["step"]
    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=2.4, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.8, ls="-", alpha=0.9)
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=1.2, ls="-", alpha=0.7)
    plt.plot(time, stats["local-cost"], label="local-cost",
             color="tab:orange", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(optimal_cost, color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.bar(time, stats["k"], width=1.,
              color="tab:gray", alpha=0.3, zorder=-10)
    plt.ylabel("k (neighborhood order)")

    plt.title(f"vns - {inst_name} \n "
              f"hit: {hit} -> best: {best_cost}, opt: {optimal_cost}")

    plt.savefig(save_fig_path, dpi=300)
    plt.close()

    pbar.update()

results_df = pd.DataFrame(results)

save_results_path = os.path.join(OUTPUT_DIR, "vnsearch-run", "results.csv")
results_df.to_csv(save_results_path)


print(f"total hits: {results_df["hit"].sum()}")
print(f"mean relative error: {results_df["relative-error"].mean()}")
print(f"mean runtime: {results_df["runtime"].mean()}")

