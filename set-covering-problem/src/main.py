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

results["instance"].extend(available_inst)

heuristic = "greedy"
results[f"{heuristic}-cost"] = []
results[f"{heuristic}-cost-re"] = []

if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic)):
    os.mkdir(path)
for inst_name in available_inst:
    inst = dl.load_instance(inst_name)

    solver = GreedySolver(instance=inst)
    log_path = os.path.join(OUTPUT_DIR, heuristic, f"{inst_name}.log")
    solver.configure_logger(path=log_path)
    sol, cost = solver.greedy_heuristic()
    sol_RE, cost_RE = solver.clean_redundant()

    results[f"{heuristic}-cost"].append(cost)
    results[f"{heuristic}-cost-re"].append(cost_RE)

heuristic = "better-greedy"
results[f"{heuristic}-cost"] = []
results[f"{heuristic}-cost-re"] = []

if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic)):
    os.mkdir(path)
for inst_name in available_inst:
    inst = dl.load_instance(inst_name)

    solver = BetterGreedySolver(instance=inst)
    log_path = os.path.join(OUTPUT_DIR, heuristic, f"{inst_name}.log")
    solver.configure_logger(path=log_path)
    sol, cost = solver.greedy_heuristic()
    sol_RE, cost_RE = solver.clean_redundant()

    results[f"{heuristic}-cost"].append(cost)
    results[f"{heuristic}-cost-re"].append(cost_RE)

results_df = pd.DataFrame.from_dict(
    results
)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "results.csv")
)

print(results_df)