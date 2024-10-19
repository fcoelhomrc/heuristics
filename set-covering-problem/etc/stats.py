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

n_inst = 42

for h in heuristics:
    for i in improvers:
        name = f"{h}-{i}"
        key1 = f"{h}-constructive-step-runtime"
        key2 = f"{h}-redundancy-elimination-runtime"
        key3 = f"{h}-{i}-improvement-step-runtime"
        agg = df[[key1, key2, key3]].sum().sum() / n_inst
        print(f"{name} -> {agg:.1f} s")


print("\nDEVIATION FROM BEST KNOWN SOLUTION")
print("=" * 20)

for h in heuristics:
    key = f"{h}-error"
    agg = df[key].mean()
    agg2 = df[key].std()
    print(f"{h} -> {agg * 100:.1f} +- {agg2 * 100:.1f} %")
    for i in improvers:
        name = f"{h}-{i}"
        key = f"{name}-error"
        agg = df[key].mean()
        agg2 = df[key].std()
        print(f"{name} -> {agg * 100:.1f} % +- {agg2 * 100:.1f} %")


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
        costs[f"{h}-{i}"] = df[key]

def wilcoxon_test(costs):
    # Get all algorithm names
    algorithms = list(costs.keys())
    size = len(algorithms)

    pvalues = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            alg1 = algorithms[i]
            alg2 = algorithms[j]
            costs1 = costs[alg1]
            costs2 = costs[alg2]

            stat, p = wilcoxon(costs1, costs2)
            pvalues[i, j] = p
    return algorithms, pvalues

plabels, pvalues = wilcoxon_test(costs)

# Display results
print("=" * 20)
print("WILCOXON PAIR-WISE COMPARISON")
alpha = 0.05

pdf = pd.DataFrame(pvalues, columns=plabels, index=plabels)

import os
PVALUES_SAVE = os.path.join(
    "..", "output", "pvalues.csv"
)
pdf.to_csv(PVALUES_SAVE)

print(pdf)


import matplotlib.pyplot as plt
import seaborn as sns

# sns.heatmap(pdf, cbar=True, annot=True)
# plt.title("Wilcoxon p-values")
# plt.show()

####
# sns.heatmap(pdf < alpha, cbar=True, annot=True, cmap="Greens")
# plt.title("Significant differences")
# plt.tight_layout()
# plt.show()
