import pandas as pd

from annealing import SimulatedAnnealing, EquilibriumStrategy, CoolingSchedule
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

calibration_instances = {
    "42": None, "51": None, "61": None, "a1": None, "b1": None, "c1": None, "d1": None,
}

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



    if inst_name in calibration_instances:
        temperatures = [1, 30, 60, 90]
        costs = []
        for initial_temperature in temperatures:
            max_iter = 30
            max_cand = 300
            Ti = initial_temperature
            Tf = 0.01
            alpha = (Tf / Ti) ** (1 / max_iter)

            inst = dl.load_instance(inst_name)
            solver = GreedySolver(instance=inst)
            solver.configure_logger(path=log_path)
            sol_greedy, cost_greedy = solver.greedy_heuristic()

            # reset instance
            inst = dl.load_instance(inst_name)
            inst.set_sol(list(sol_greedy))
            inst.set_state(State.SOLVED)

            sim_annealing = SimulatedAnnealing(
                instance=inst,
                max_candidates=max_cand, max_iterations=max_iter,
                initial_temperature=Ti, final_temperature=Tf,
                cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": alpha},
                equilibrium_strategy=EquilibriumStrategy.STATIC
            )
            best, best_cost = sim_annealing.run()
            costs.append(best_cost)
        costs = np.array(costs)
        fine_tune_temperature = temperatures[costs.argmin()]
        delta_temperature = 5
        n_samples = 5
        temperatures = np.linspace(fine_tune_temperature - delta_temperature,
                                   fine_tune_temperature + delta_temperature, n_samples)
        costs = []
        for initial_temperature in temperatures:
            max_iter = 30
            max_cand = 300
            Ti = initial_temperature
            Tf = 0.01
            alpha = (Tf / Ti) ** (1 / max_iter)

            inst = dl.load_instance(inst_name)
            solver = GreedySolver(instance=inst)
            solver.configure_logger(path=log_path)
            sol_greedy, cost_greedy = solver.greedy_heuristic()

            # reset instance
            inst = dl.load_instance(inst_name)
            inst.set_sol(list(sol_greedy))
            inst.set_state(State.SOLVED)

            sim_annealing = SimulatedAnnealing(
                instance=inst,
                max_candidates=max_cand, max_iterations=max_iter,
                initial_temperature=Ti, final_temperature=Tf,
                cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": alpha},
                equilibrium_strategy=EquilibriumStrategy.STATIC
            )
            best, best_cost = sim_annealing.run()
            costs.append(best_cost)
        costs = np.array(costs)
        chosen_temperature = temperatures[costs.argmin()]
        calibration_instances[inst_name] = chosen_temperature

    # get calibration result
    char = inst_name[0]
    if char == "4":
        Ti = calibration_instances["42"]
    elif char == "5":
        Ti = calibration_instances["51"]
    elif char == "6":
        Ti = calibration_instances["61"]
    elif char == "a":
        Ti = calibration_instances["a1"]
    elif char == "b":
        Ti = calibration_instances["b1"]
    elif char == "c":
        Ti = calibration_instances["c1"]
    elif char == "d":
        Ti = calibration_instances["d1"]
    else:
        Ti = 30

    max_iter = 100
    max_cand = 1000
    Tf = 0.01
    alpha = (Tf / Ti) ** (1 / max_iter)

    sim_annealing = SimulatedAnnealing(
        instance=inst,
        max_candidates=max_cand, max_iterations=max_iter,
        initial_temperature=Ti, final_temperature=Tf,
        cooling_schedule=CoolingSchedule.GEOMETRIC, cooling_params={"rate": alpha},
        equilibrium_strategy=EquilibriumStrategy.STATIC
    )
    sim_annealing.configure_logger(log_path)

    results["initial-temperature"] = Ti
    results["final-temperature"] = Tf
    results["cooling-rate"] = alpha
    print(f"Ti {Ti} -> Tf {Tf} (alpha = {alpha:.5f} -> {max_iter} iter)")

    # run annealing algorithm and hope for the best
    best, best_cost = sim_annealing.run()
    optimal_cost = sim_annealing.get_cost_optimal()
    hit = optimal_cost == best_cost
    stats = sim_annealing.get_stats()

    results["runtime"].append(perf_counter() - start)

    init_size = len(sim_annealing.initial_sol)
    best_size = len(best)

    init_cost = sim_annealing.do_compute_cost(sim_annealing.initial_sol)
    relative_error = abs(optimal_cost - best_cost) / optimal_cost

    results["init-size"].append(init_size)
    results["best-size"].append(best_size)
    results["init-cost"].append(init_cost)
    results["best-cost"].append(best_cost)
    results["optimal-cost"].append(optimal_cost)
    results["hit"].append(hit)
    results["relative-error"].append(relative_error)

    assert sim_annealing.is_complete_solution(best)  # sanity check

    import matplotlib.pyplot as plt
    save_fig_path = os.path.join(OUTPUT_DIR, "annealing-run", f"{inst_name}.png")
    time = stats["step"]

    plt.figure()
    plt.plot(time, stats["best-cost"], label="best-cost",
             color="tab:green", lw=2.4, ls="-")
    plt.plot(time, stats["current-cost"], label="current-cost",
             color="k", lw=1.4, ls="-", alpha=0.9)
    plt.plot(time, stats["candidate-cost"], label="candidate-cost",
             color="tab:blue", lw=0.8, ls="-", alpha=0.7)
    plt.axhline(inst.get_best_known_sol_cost(), color="k", lw=2.0, ls="--")

    plt.xlabel("step")
    plt.ylabel("costs")
    plt.legend()

    twinx = plt.twinx()
    twinx.plot(time, np.asarray(stats["temperature"]), label="temperature",
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


print(f"calibration: {calibration_instances}")
print(f"total hits: {results_df["hit"].sum()}")
print(f"mean relative error: {results_df["relative-error"].mean()}")
print(f"mean runtime: {results_df["runtime"].mean()}")

