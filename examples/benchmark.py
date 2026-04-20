import time
import random
import numpy as np
import fast_thompson as ft
from collections import defaultdict


# -----------------------------
# NumPy baseline sampler
# -----------------------------
def numpy_sample(items, k, forbidden_set):
    # filter forbidden
    filtered = [it for it in items if it[0] not in forbidden_set]

    scores = []

    for item_id, s, f in filtered:
        score = np.random.beta(s + 1, f + 1)
        scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [item_id for item_id, _ in scores[:k]]


# -----------------------------
# Generate dataset
# -----------------------------
def generate_items(n):
    items = []
    for i in range(n):
        s = random.randint(1, 50)
        f = random.randint(1, 50)
        items.append((i, s, f))
    return items


# -----------------------------
# Benchmark runner
# -----------------------------
def benchmark_cpp(model, forbiddens_list, k, runs):
    start = time.perf_counter()

    for forbidden in forbiddens_list:
        model.sample(k, forbidden)

    return time.perf_counter() - start


def benchmark_numpy(items, forbiddens_list, k, runs):
    start = time.perf_counter()

    for forbidden in forbiddens_list:
        numpy_sample(items, k, forbidden)

    return time.perf_counter() - start


# -----------------------------
# Main test
# -----------------------------
def main():
    N = 5000
    K_min, K_max = 100, 300
    RUNS = 1000
    TOP_K = 10

    print("Generating dataset...")

    items_raw = generate_items(N)

    cpp_items = [
        ft.Item(i, s, f) for i, s, f in items_raw
    ]

    model = ft.Thompson(cpp_items)

    # pre-generate forbidden sets
    forbiddens_list = []
    for _ in range(RUNS):
        k = random.randint(K_min, K_max)
        forbiddens = random.sample(range(N), k)
        forbiddens_list.append(forbiddens)

    print("Running C++ benchmark...")
    cpp_time = benchmark_cpp(model, forbiddens_list, TOP_K, RUNS)

    print("Running NumPy benchmark...")
    np_time = benchmark_numpy(items_raw, forbiddens_list, TOP_K, RUNS)

    print("\n--- RESULTS ---")
    print(f"C++ Thompson: {cpp_time:.4f}s")
    print(f"NumPy baseline: {np_time:.4f}s")
    print(f"Speedup: {np_time / cpp_time:.2f}x")


if __name__ == "__main__":
    main()