import pandas as pd

from annealing import SimulatedAnnealing, EquilibriumStrategy, CoolingSchedule, MoveType
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
        "initial-temperature": [],
        "final-temperature": [],
        "cooling-rate": [],
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
if not os.path.exists(path := os.path.join(OUTPUT_DIR, "annealing-run")):
    os.mkdir(path)

pbar = tqdm(total=len(available_inst),
            desc="instance", position=0, leave=False)

for inst_name in available_inst:
    start = perf_counter()

    log_path = os.path.join(OUTPUT_DIR, "annealing-test-run", f"{inst_name}.log")
    dl = DataLoader()
    inst = dl.load_instance(inst_name)
    solver = GreedySolver(instance=inst)
    solver.configure_logger(path=log_path)
    sol_greedy, cost_greedy = solver.greedy_heuristic()

    # reset instance
    inst = dl.load_instance(inst_name)
    inst.set_sol(list(sol_greedy))
    inst.set_state(State.SOLVED)

    max_iter = 1_000  # not used in EquilibriumStrategy.EXHAUSTIVE
    max_cand = 10 * inst.get_cols()  # not used in EquilibriumStrategy.EXHAUSTIVE
    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=max_cand, max_iterations=max_iter,  # dummy parameters
        initial_temperature=5.0, final_temperature=1.,  # dummy parameters
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": 0.99}
    )
    sim_annealing.configure_logger(log_path)

    # estimate hyper parameters
    init_temp, final_temp, cooling = sim_annealing.dry_run(
        max_iterations=max_iter
    )

    sim_annealing.set_hyper_parameters(
        initial_temperature=init_temp,
        final_temperature=final_temp,
        cooling_params=cooling
    )

    results["initial-temperature"] = init_temp
    results["final-temperature"] = final_temp
    results["cooling-rate"] = cooling["rate"]

    # solve instance
    sim_annealing.run()

    results["runtime"].append(perf_counter() - start)

    best = sim_annealing.get_best()
    stats = sim_annealing.get_stats()

    init_size = len(sim_annealing.initial_sol)
    best_size = len(best)

    sim_annealing.inst.set_sol(sim_annealing.initial_sol)
    init_cost = sim_annealing.inst.get_sol_cost()

    sim_annealing.inst.set_sol(best)
    best_cost = sim_annealing.inst.get_sol_cost()

    optimal_cost = sim_annealing.inst.get_best_known_sol_cost()
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

    import matplotlib.pyplot as plt
    save_fig_path = os.path.join(OUTPUT_DIR, "annealing-run", f"{inst_name}.png")
    time = stats["step"]

    plt.figure()
    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=1.5, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.2, ls="-")
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(inst.get_best_known_sol_cost(), color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, np.asarray(stats["temperature"]) / sim_annealing.initial_temperature, label="temperature",
               color="tab:orange", ls="-", lw=0.8)
    plt.ylabel("temperature")

    plt.title(f"simulated annealing - instance {inst_name} \n "
              f"hit: {hit} -> best: {best_cost}, opt: {optimal_cost}")
    plt.savefig(save_fig_path, dpi=300)
    plt.close()

    pbar.update()

results_df = pd.DataFrame(results)

save_results_path = os.path.join(OUTPUT_DIR, "annealing-run", "results.csv")
results_df.to_csv(save_results_path)

print(f"total hits: {results_df["hit"].sum()}")
print(f"mean relative error: {results_df["relative-error"].mean()}")
print(f"mean runtime: {results_df["runtime"].mean()}")

