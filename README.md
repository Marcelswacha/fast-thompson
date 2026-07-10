# fast-thompson

Fast Thompson Sampling for multi-armed bandits, implemented in C++ (AVX2) with Python bindings.

Each item keeps a Beta(successes + 1, failures + 1) posterior. Calling `sample` draws a score from every posterior and returns the top-k items. Low-count items are sampled exactly (Marsaglia–Tsang), while high-count, low-skew posteriors use a fast normal approximation. Randomness comes from a vectorized xoshiro256++ generator with batched, buffer-based sampling — roughly 2x faster than the equivalent numpy code.

## Install

```bash
pip install git+https://github.com/MarcelSwacha/fast-thompson.git
```

## Usage

```python
import fast_thompson as ft

items = [
    ft.Item(1, 10, 5),   # id, successes, failures
    ft.Item(2, 3, 8),
    ft.Item(3, 20, 2),
]

model = ft.Thompson(items, seed=42)  # seed is optional
top = model.sample(2, [])            # top-k, list of forbidden ids

for r in top:
    print(r.id, r.score)
```

## Running tests

```bash
pip install .
pip install numpy scipy pytest
pytest tests/ -v
```

## Requirements

- Python >= 3.8
- x86-64 CPU with AVX2