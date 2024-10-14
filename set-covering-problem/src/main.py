from utils import DataLoader, Instance, State
from solvers import *
import pandas as pd
import os
from tqdm import tqdm

OUTPUT_DIR = "../output"

results = dict(
    {
        "instance": []
    }
)

dl = DataLoader()
available_inst = dl.list_instances()

best_known_solutions =  [512, 516, 494, 512, 560, 430, 492, 5441, 253, 302, 226,
                         242, 211, 213, 293, 288, 279, 138, 146, 145, 131, 161,
                         253, 252, 232, 234, 236, 69, 76, 80, 79, 72, 227, 219,
                         243, 219, 215, 60, 66, 72, 62, 61]

results["instance"].extend(available_inst)
results[f"optimal-cost"] = best_known_solutions

heuristics = {
    "greedy": GreedySolver,
    "random-greedy": RandomGreedySolver,
    "priority-greedy": PriorityGreedySolver
}


for heuristic_name, heuristic in tqdm(heuristics.items()):
    results[f"{heuristic_name}-cost"] = []
    results[f"{heuristic_name}-cost-re"] = []

    results[f"{heuristic_name}-error"] = []
    results[f"{heuristic_name}-error-re"] = []

    results[f"{heuristic_name}-re-improves"] = []

    for step, runtime in heuristic(Instance(1, 1)).get_elapsed_times().items():
        results[f"{heuristic_name}-{step}-runtime"] = []

    if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic_name)):
        os.mkdir(path)

    for inst_name, cost_best in zip(available_inst, best_known_solutions):
        inst = dl.load_instance(inst_name)

        solver = heuristic(instance=inst)
        log_path = os.path.join(OUTPUT_DIR, heuristic_name, f"{inst_name}.log")
        solver.configure_logger(path=log_path)
        sol, cost = solver.greedy_heuristic()
        sol_RE, cost_RE = solver.redundancy_elimination()

        results[f"{heuristic_name}-cost"].append(cost)
        results[f"{heuristic_name}-error"].append(np.abs(cost - cost_best)/cost_best)
        results[f"{heuristic_name}-cost-re"].append(cost_RE)
        results[f"{heuristic_name}-error-re"].append(np.abs(cost_RE - cost_best) / cost_best)
        results[f"{heuristic_name}-re-improves"].append(cost_RE < cost)

        for step, runtime in solver.get_elapsed_times().items():
            results[f"{heuristic_name}-{step}-runtime"].append(runtime)

results_df = pd.DataFrame.from_dict(
    results
)

for heuristic in heuristics:
    col = f"{heuristic}-error"
    print(f"{col}: {results_df[col].mean()} +- {2*results_df[col].std()} [{results_df[col].min()}, {results_df[col].max()}]")

    col = f"{heuristic}-error-re"
    print(f"{col}: {results_df[col].mean()} +- {2*results_df[col].std()} [{results_df[col].min()}, {results_df[col].max()}]")


results_df.to_csv(
    os.path.join(OUTPUT_DIR, "results.csv")
)



# print(results_df)