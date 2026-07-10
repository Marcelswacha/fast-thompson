import time
import random

import numpy as np
import fast_thompson as ft


# -----------------------------
# NumPy baseline 1: per-item Python loop
# (typical "straightforward" implementation)
# -----------------------------
def numpy_sample_loop(items, k, forbidden):
    forbidden_set = set(forbidden)  # request-time conversion, part of the cost
    filtered = [it for it in items if it[0] not in forbidden_set]
    scores = []
    for item_id, s, f in filtered:
        score = np.random.beta(s + 1, f + 1)
        scores.append((item_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scores[:k]]


# -----------------------------
# NumPy baseline 2: fully vectorized
# (the fastest pure-numpy implementation: one batched beta draw,
#  boolean-mask filtering, O(n) top-k via argpartition)
# -----------------------------
def numpy_sample_vec(ids, alphas, betas, k, forbidden, rng):
    mask = np.ones(len(ids), dtype=bool)
    mask[forbidden] = False  # accepts the raw Python list; conversion cost included

    draws = rng.beta(alphas[mask], betas[mask])

    idx = np.argpartition(draws, -k)[-k:]
    return ids[mask][idx]


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
# Benchmark runners
# -----------------------------
def benchmark_cpp(model, forbiddens_list, k):
    start = time.perf_counter()
    for forbidden in forbiddens_list:
        model.sample(k, forbidden)
    return time.perf_counter() - start


def benchmark_numpy_loop(items, forbiddens_list, k):
    start = time.perf_counter()
    for forbidden in forbiddens_list:
        numpy_sample_loop(items, k, forbidden)
    return time.perf_counter() - start


def benchmark_numpy_vec(items, forbiddens_list, k):
    # Model state (precomputed once, like the Thompson object) stays outside
    # the timed loop. The forbidden lists do NOT: they arrive per-request.
    ids = np.array([i for i, _, _ in items], dtype=np.int64)
    alphas = np.array([s + 1.0 for _, s, _ in items])
    betas = np.array([f + 1.0 for _, _, f in items])
    rng = np.random.default_rng()

    start = time.perf_counter()
    for forbidden in forbiddens_list:
        numpy_sample_vec(ids, alphas, betas, k, forbidden, rng)
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

    cpp_items = [ft.Item(i, s, f) for i, s, f in items_raw]
    model = ft.Thompson(cpp_items)

    # pre-generate forbidden sets
    forbiddens_list = []
    for _ in range(RUNS):
        k = random.randint(K_min, K_max)
        forbiddens_list.append(random.sample(range(N), k))

    print("Running C++ benchmark...")
    cpp_time = benchmark_cpp(model, forbiddens_list, TOP_K)

    print("Running NumPy (loop) benchmark...")
    np_loop_time = benchmark_numpy_loop(items_raw, forbiddens_list, TOP_K)

    print("Running NumPy (vectorized) benchmark...")
    np_vec_time = benchmark_numpy_vec(items_raw, forbiddens_list, TOP_K)

    print("\n--- RESULTS ---")
    print(f"C++ Thompson:        {cpp_time:.4f}s   ({cpp_time / RUNS * 1e3:.3f} ms/call)")
    print(f"NumPy (loop):        {np_loop_time:.4f}s   ({np_loop_time / RUNS * 1e3:.3f} ms/call)")
    print(f"NumPy (vectorized):  {np_vec_time:.4f}s   ({np_vec_time / RUNS * 1e3:.3f} ms/call)")
    print(f"\nSpeedup vs loop:        {np_loop_time / cpp_time:.2f}x")
    print(f"Speedup vs vectorized:  {np_vec_time / cpp_time:.2f}x")


if __name__ == "__main__":
    main()