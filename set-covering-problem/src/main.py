from utils import DataLoader, Instance, State
from solvers import *
from improvement import *
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

best_known_solutions =  [
    512, 516, 494, 512, 560, 430, 492, 641, # 4.X
    253, 302, 226, 242, 211, 213, 293, 288, 279, # 5.X
    138, 146, 145, 131, 161, # 6.X
    253, 252, 232, 234, 236, # A.X
    69, 76, 80, 79, 72, # B.X
    227, 219, 243, 219, 215, # C.X
    60, 66, 72, 62, 61 # D.X
]

results["instance"].extend(available_inst)
results[f"optimal-cost"] = best_known_solutions

heuristics = {
    "greedy": GreedySolver,
    "random-greedy": RandomGreedySolver,
    "priority-greedy": PriorityGreedySolver,
    "greedy-re": GreedySolver
}

local_search_strategies = {
    "first-search": SearchStrategy.FIRST,
    "best-search": SearchStrategy.BEST
}

pbar = tqdm(total=len(heuristics) * len(local_search_strategies),
            desc="Algorithm")
pbar2 = tqdm(total=42, desc="Instance")

for heuristic_name, heuristic in heuristics.items():
    for ls_strategy_name, ls_strategy in local_search_strategies.items():
        results[f"{heuristic_name}-cost"] = []
        results[f"{heuristic_name}-error"] = []

        results[f"{heuristic_name}-{ls_strategy_name}-cost"] = []
        results[f"{heuristic_name}-{ls_strategy_name}-error"] = []

        for step, runtime in heuristic(Instance(1, 1)).get_elapsed_times().items():
            results[f"{heuristic_name}-{step}-runtime"] = []

        for step, runtime in LocalSearch(
                heuristic=heuristic(Instance(1, 1))
        ).get_elapsed_times().items():
            results[f"{heuristic_name}-{ls_strategy_name}-{step}-runtime"] = []

        if not os.path.exists(path := os.path.join(OUTPUT_DIR, heuristic_name)):
            os.mkdir(path)

        for inst_name, cost_best in zip(available_inst, best_known_solutions):
            inst = dl.load_instance(inst_name)

            solver = heuristic(instance=inst)
            log_path = os.path.join(OUTPUT_DIR, heuristic_name, f"{inst_name}.log")
            solver.configure_logger(path=log_path)
            sol, cost = solver.greedy_heuristic()
            solver.inst.set_state(State.SOLVED)

            if "-re" in heuristic_name:
                sol, cost = solver.redundancy_elimination()

            results[f"{heuristic_name}-cost"].append(cost)
            results[f"{heuristic_name}-error"].append(np.abs(cost - cost_best)/cost_best)

            for step, runtime in solver.get_elapsed_times().items():
                results[f"{heuristic_name}-{step}-runtime"].append(runtime)

            local_search = LocalSearch(heuristic=solver, strategy=ls_strategy)
            sol_improved, cost_improved = local_search.run()

            results[f"{heuristic_name}-{ls_strategy_name}-cost"].append(cost_improved)
            results[f"{heuristic_name}-{ls_strategy_name}-error"].append(np.abs(cost_improved - cost_best) / cost_best)

            for step, runtime in local_search.get_elapsed_times().items():
                results[f"{heuristic_name}-{ls_strategy_name}-{step}-runtime"].append(runtime)

            pbar2.update()
        pbar.update()
        pbar2.reset()

results_df = pd.DataFrame.from_dict(
    results
)

for heuristic in heuristics:
    col = f"{heuristic}-error"
    print(f"{col}: {results_df[col].mean()} +- {2*results_df[col].std()} [{results_df[col].min()}, {results_df[col].max()}]")

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "results.csv")
)

# print(results_df)