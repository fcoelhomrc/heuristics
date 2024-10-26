import pandas as pd

from grasp import Grasp, GraspVND
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

OUTPUT_DIR = "../output"
if not os.path.exists(path := os.path.join(OUTPUT_DIR, "grasp-vnd-run")):
    os.mkdir(path)

pbar = tqdm(total=len(available_inst),
            desc="instance", position=0, leave=False)


for inst_name in available_inst:
    start = perf_counter()
    results["instance"].append(inst_name)

    log_path = os.path.join(OUTPUT_DIR, "grasp-vnd-run", f"{inst_name}.log")
    dl = DataLoader()
    inst = dl.load_instance(inst_name)

    K = 4
    alpha = K / inst.get_cols()
    n_solutions = 16
    grasp = GraspVND(instance=inst,
                     alpha=alpha, n_solutions=n_solutions)

    grasp.configure_logger(log_path)
    max_iter = 100
    grasp.max_iter = max_iter
    max_time = 0.5 * 60  # around 1h for 42 instances
    grasp.max_runtime = max_time

    # solve instance
    grasp.run()

    results["runtime"].append(perf_counter() - start)

    initial, init_cost = grasp.get_best()  # initial solutions
    best, best_cost = grasp.get_optimized_best()  # after optimization

    init_size = len(initial)
    best_size = len(best)

    optimal_cost = inst.get_best_known_sol_cost()
    hit = (best_cost == optimal_cost)
    relative_error = abs(optimal_cost - best_cost) / optimal_cost

    results["init-size"].append(init_size)
    results["best-size"].append(best_size)
    results["init-cost"].append(init_cost)
    results["best-cost"].append(best_cost)
    results["optimal-cost"].append(optimal_cost)
    results["hit"].append(hit)
    results["relative-error"].append(relative_error)

    inst.set_sol(best)
    assert inst.check_sol_coverage()  # sanity check
    pbar.update()

results_df = pd.DataFrame(results)

save_results_path = os.path.join(OUTPUT_DIR, "grasp-vnd-run", "results.csv")
results_df.to_csv(save_results_path)

print(f"total hits: {results_df["hit"].sum()}")
print(f"mean relative error: {results_df["relative-error"].mean()}")
print(f"mean runtime: {results_df["runtime"].mean()}")

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.barplot(data=results_df, x="instance", y="hit", ax=axes[0, 0])
sns.barplot(data=results_df, x="instance", y="relative-error", ax=axes[0, 1])
df_melted = results_df.melt(id_vars='instance', value_vars=['init-cost', 'best-cost', 'optimal-cost'],
                            var_name='stage', value_name='cost')
sns.barplot(data=df_melted, x="instance", y="cost", hue='stage', ax=axes[1, 0])
sns.barplot(data=results_df, x="instance", y="runtime", ax=axes[1, 1])
plt.tight_layout()
plt.show()
