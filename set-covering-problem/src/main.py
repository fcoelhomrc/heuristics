from utils import DataLoader, Instance, State
from solvers import *
import pandas as pd
import os

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

heuristic = "greedy"
results[f"{heuristic}-cost"] = []
results[f"{heuristic}-cost-re"] = []

results[f"{heuristic}-error"] = []
results[f"{heuristic}-error-re"] = []

results[f"{heuristic}-re-improves"] = []

if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic)):
    os.mkdir(path)
for inst_name, cost_best in zip(available_inst, best_known_solutions):
    inst = dl.load_instance(inst_name)

    solver = GreedySolver(instance=inst)
    log_path = os.path.join(OUTPUT_DIR, heuristic, f"{inst_name}.log")
    solver.configure_logger(path=log_path)
    sol, cost = solver.greedy_heuristic()
    sol_RE, cost_RE = solver.clean_redundant()

    results[f"{heuristic}-cost"].append(cost)
    results[f"{heuristic}-error"].append((cost - cost_best)/cost_best)
    results[f"{heuristic}-cost-re"].append(cost_RE)
    results[f"{heuristic}-error-re"].append((cost_RE - cost_best) / cost_best)
    results[f"{heuristic}-re-improves"].append(cost_RE < cost)


heuristic = "better-greedy"
results[f"{heuristic}-cost"] = []
results[f"{heuristic}-cost-re"] = []

results[f"{heuristic}-error"] = []
results[f"{heuristic}-error-re"] = []

results[f"{heuristic}-re-improves"] = []

if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic)):
    os.mkdir(path)
for inst_name, cost_best in zip(available_inst, best_known_solutions):
    inst = dl.load_instance(inst_name)

    solver = BetterGreedySolver(instance=inst)
    log_path = os.path.join(OUTPUT_DIR, heuristic, f"{inst_name}.log")
    solver.configure_logger(path=log_path)
    sol, cost = solver.greedy_heuristic()
    sol_RE, cost_RE = solver.clean_redundant()

    results[f"{heuristic}-cost"].append(cost)
    results[f"{heuristic}-error"].append((cost - cost_best)/cost_best)
    results[f"{heuristic}-cost-re"].append(cost_RE)
    results[f"{heuristic}-error-re"].append((cost_RE - cost_best) / cost_best)
    results[f"{heuristic}-re-improves"].append(cost_RE < cost)

heuristic = "random-greedy"
results[f"{heuristic}-cost"] = []
results[f"{heuristic}-cost-re"] = []

results[f"{heuristic}-error"] = []
results[f"{heuristic}-error-re"] = []

results[f"{heuristic}-re-improves"] = []

if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic)):
    os.mkdir(path)
for inst_name, cost_best in zip(available_inst, best_known_solutions):
    inst = dl.load_instance(inst_name)

    solver = RandomGreedySolver(instance=inst)
    log_path = os.path.join(OUTPUT_DIR, heuristic, f"{inst_name}.log")
    solver.configure_logger(path=log_path)
    sol, cost = solver.greedy_heuristic()
    sol_RE, cost_RE = solver.clean_redundant()

    results[f"{heuristic}-cost"].append(cost)
    results[f"{heuristic}-error"].append((cost - cost_best)/cost_best)
    results[f"{heuristic}-cost-re"].append(cost_RE)
    results[f"{heuristic}-error-re"].append((cost_RE - cost_best) / cost_best)
    results[f"{heuristic}-re-improves"].append(cost_RE < cost)

results_df = pd.DataFrame.from_dict(
    results
)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "results.csv")
)

for col in results_df.columns:
    if "improves" in str(col):
        print(col, results_df[col].sum())

# print(results_df)