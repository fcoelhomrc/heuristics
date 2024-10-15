import pandas as pd


df = pd.read_csv("../output/results.csv")

heuristics = [
    "greedy", "random-greedy", "priority-greedy", "greedy-re"
]
improvers = [
    "best-search", "first-search"
]

cols = list(df.columns)


print("RUNTIME")
print("=" * 20)

for h in heuristics:
    for i in improvers:
        name = f"{h}-{i}"
        key1 = f"{h}-constructive-step-runtime"
        key2 = f"{h}-redundancy-elimination-runtime"
        key3 = f"{h}-{i}-improvement-step-runtime"
        agg = df[[key1, key2, key3]].sum().sum()
        print(f"{name} -> {agg:.1f} s")


print("\nDEVIATION FROM BEST KNOWN SOLUTION")
print("=" * 20)

for h in heuristics:
    for i in improvers:
        name = f"{h}-{i}"
        key = f"{name}-error"
        agg = df[key].mean()
        print(f"{name} -> {agg * 100:.1f} %")


print("\nINSTANCES WHERE LOCAL SEARCH HELPS")
print("=" * 20)

for h in heuristics:
    for i in improvers:
        name = f"{h}-{i}"
        key1 = f"{h}-cost"
        key2 = f"{name}-cost"
        local_search_improves = (df[key1] > df[key2]).sum() / len(df)
        print(f"{name} -> {local_search_improves * 100:.1f} %")


# Chat-gpt generated
# Wilcoxon pair-wise tests
import numpy as np
from scipy.stats import wilcoxon
import itertools

costs = {}
for h in heuristics:
    for i in improvers:
        key = f"{h}-{i}-cost"
        if "best" not in i:
            continue
        costs[f"{h}-{i}"] = df[key]

def wilcoxon_test(costs):
    # Get all algorithm names
    algorithms = list(costs.keys())

    # Generate all pairs of algorithms to compare
    pairs = list(itertools.combinations(algorithms, 2))

    # Initialize a dictionary to store Wilcoxon test results
    wilcoxon_results = {}

    for alg1, alg2 in pairs:
        # Perform Wilcoxon signed-rank test for paired data
        stat, p_value = wilcoxon(costs[alg1], costs[alg2])
        wilcoxon_results[(alg1, alg2)] = (stat, p_value)

    return wilcoxon_results

results = wilcoxon_test(costs)

# Display results
print("=" * 20)
for pair, (stat, p_value) in results.items():
    print(f"Wilcoxon test between {pair[0]} and {pair[1]}: statistic={stat}, p-value={p_value}")
