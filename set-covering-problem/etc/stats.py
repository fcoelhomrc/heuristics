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

