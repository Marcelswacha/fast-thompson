import fast_thompson as ft
from collections import defaultdict
import numpy as np


# -----------------------------
# NumPy Thompson sampler (baseline)
# -----------------------------
def numpy_sample(items, k):
    # items = [(id, success, failure), ...]
    scores = []

    for item_id, s, f in items:
        score = np.random.beta(s + 1, f + 1)
        scores.append((item_id, score))

    # sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return [item_id for item_id, _ in scores[:k]]


# -----------------------------
# Run simulation
# -----------------------------
def run_simulation(sample_fn, items, iterations, k):
    counts = defaultdict(int)

    for _ in range(iterations):
        result = sample_fn(k)
        for r in result:
            counts[r] += 1

    return counts


def print_results(label, counts, iterations, k):
    print(f"\n{label}:\n")
    for item_id in sorted(counts.keys()):
        print(
            f"Item {item_id}: {counts[item_id]} "
            f"({counts[item_id] / (iterations * k):.3%})"
        )


# -----------------------------
# Main
# -----------------------------
def main():
    items_cpp = [
        ft.Item(1, 10, 5),
        ft.Item(2, 3, 8),
        ft.Item(3, 20, 2),
        ft.Item(4, 7, 7),
        ft.Item(5, 15, 10),
    ]

    items_np = [(i.id, i.successes, i.failures) for i in items_cpp]

    model = ft.Thompson(items_cpp)

    forbidden = []

    iterations = 5000
    k = 3

    # -----------------------------
    # C++ sampler wrapper
    # -----------------------------
    def cpp_sample(k):
        return [r.id for r in model.sample(k, forbidden)]

    # -----------------------------
    # NumPy sampler wrapper
    # -----------------------------
    def np_sample(k):
        return numpy_sample(items_np, k)

    cpp_counts = run_simulation(cpp_sample, items_cpp, iterations, k)
    np_counts = run_simulation(np_sample, items_np, iterations, k)

    print_results("C++ Thompson sampler", cpp_counts, iterations, k)
    print_results("NumPy beta-rank baseline", np_counts, iterations, k)


if __name__ == "__main__":
    main()