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

## C++ benchmark (for profiling)

```bash
# one-shot: build + perf stat + perf record
./scripts/profile.sh [workload] [stat|record|both] [repeat]

# or manually:
cmake -B build -DBUILD_BENCHMARK=ON -DBUILD_PYTHON_MODULE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ft_bench [mixed|exact|approx|nofb|big|gen|all] [repeat]

# call graphs
perf record -g ./build/ft_bench mixed
perf report

# counters over the measured region only (generation/warmup excluded)
mkfifo /tmp/perf_ctl /tmp/perf_ack
PERF_CTL_FIFO=/tmp/perf_ctl PERF_ACK_FIFO=/tmp/perf_ack \
perf stat -d --delay=-1 --control fifo:/tmp/perf_ctl,/tmp/perf_ack \
    -- ./build/ft_bench mixed
```

### perf permissions

If perf fails with a permission error, check `kernel.perf_event_paranoid`
(hardened distros set it to 3 or 4, which blocks unprivileged perf entirely):

```bash
cat /proc/sys/kernel/perf_event_paranoid

# allow user-space profiling for this session (2 = own processes, no kernel)
sudo sysctl kernel.perf_event_paranoid=2

# make it persistent
echo 'kernel.perf_event_paranoid = 2' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

Use `1` if you also want kernel-side samples in `perf record` call graphs,
or run the profiling command under `sudo` without changing the setting.