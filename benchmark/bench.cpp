// Standalone C++ benchmark for fast-thompson.
//
// Build:
//   cmake -B build -DBUILD_BENCHMARK=ON -DBUILD_PYTHON_MODULE=OFF -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//   ./build/ft_bench [workload] [repeat]
//     workload: mixed|exact|approx|nofb|big|gen|all
//     repeat:   multiplier on the measured loop (default 1); use e.g. 50
//               so perf counters cover a few seconds instead of ~70 ms
//
// Profile (call graphs):
//   perf record -g ./build/ft_bench mixed
//   perf report
//
// Counters over ONLY the measured region (dataset generation and warmup
// excluded) via perf's control FIFO -- counters start disabled and the
// benchmark toggles them around the timed loops:
//
//   mkfifo /tmp/perf_ctl /tmp/perf_ack
//   PERF_CTL_FIFO=/tmp/perf_ctl PERF_ACK_FIFO=/tmp/perf_ack \
//   perf stat -d --delay=-1 --control fifo:/tmp/perf_ctl,/tmp/perf_ack \
//       -- ./build/ft_bench mixed
//
// (Crude alternative without FIFOs: perf stat -D 500 ./build/ft_bench mixed,
//  but the 500 ms guess is machine-dependent; the FIFO method is exact.)
//
// The workloads are split into separate functions so perf attributes samples
// cleanly (end-to-end mixed/exact/approx, forbidden overhead, and raw
// generator throughput).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "Thompson.hpp"
#include "Uniform.hpp"
#include "Ziggurat.hpp"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Global sink defeats dead-code elimination.
static volatile double g_sink = 0.0;

// perf stat control: toggles counters around measured regions only.
// No-op unless PERF_CTL_FIFO / PERF_ACK_FIFO are set (see header comment).
class PerfCtl {
public:
    PerfCtl() {
        const char* ctl = std::getenv("PERF_CTL_FIFO");
        const char* ack = std::getenv("PERF_ACK_FIFO");
        if (ctl) _ctl = ::open(ctl, O_WRONLY);
        if (ack) _ack = ::open(ack, O_RDONLY);
    }
    ~PerfCtl() {
        if (_ctl >= 0) ::close(_ctl);
        if (_ack >= 0) ::close(_ack);
    }
    void enable()  { command("enable\n"); }
    void disable() { command("disable\n"); }

private:
    int _ctl = -1;
    int _ack = -1;

    void command(const char* cmd) {
        if (_ctl < 0) return;
        if (::write(_ctl, cmd, std::strlen(cmd)) < 0) return;
        if (_ack >= 0) {
            char buf[8];
            (void)!::read(_ack, buf, sizeof(buf));   // wait for "ack\n"
        }
    }
};

static PerfCtl& perf_ctl() {
    static PerfCtl ctl;
    return ctl;
}

struct Workload {
    std::vector<Item> items;
    std::vector<std::vector<uint32_t>> forbiddens;
};

static Workload make_workload(size_t n, int s_min, int s_max,
                              size_t runs, size_t fb_min, size_t fb_max,
                              uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> counts(s_min, s_max);

    Workload w;
    w.items.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        w.items.emplace_back(i, counts(gen), counts(gen));
    }

    std::uniform_int_distribution<size_t> fb_size(fb_min, fb_max);
    std::vector<uint32_t> pool(n);
    for (uint32_t i = 0; i < n; ++i) pool[i] = i;

    w.forbiddens.resize(runs);
    for (auto& fb : w.forbiddens) {
        const size_t k = fb_size(gen);
        // partial Fisher-Yates: k distinct ids
        for (size_t j = 0; j < k; ++j) {
            std::uniform_int_distribution<size_t> pick(j, n - 1);
            std::swap(pool[j], pool[pick(gen)]);
        }
        fb.assign(pool.begin(), pool.begin() + k);
    }
    return w;
}

static void bench_sample(const char* name, const Workload& w,
                         size_t top_k, size_t repeat = 1,
                         size_t warmup = 50) {
    Thompson model(w.items, /*seed=*/42);

    size_t exact = 0;
    for (const auto& it : w.items) {
        if (!it.isApprox()) ++exact;
    }

    for (size_t r = 0; r < warmup && r < w.forbiddens.size(); ++r) {
        for (const auto& res : model.sample(top_k, w.forbiddens[r]))
            g_sink += res.score;
    }

    perf_ctl().enable();
    const auto t0 = Clock::now();
    for (size_t rep = 0; rep < repeat; ++rep) {
        for (const auto& fb : w.forbiddens) {
            for (const auto& res : model.sample(top_k, fb))
                g_sink += res.score;
        }
    }
    const double ms = ms_since(t0);
    perf_ctl().disable();
    const double per_call =
        ms / static_cast<double>(w.forbiddens.size() * repeat);

    std::printf("%-28s %8.4f ms/call   (%zu items: %zu exact / %zu approx, top-%zu)\n",
                name, per_call, w.items.size(), exact,
                w.items.size() - exact, top_k);
}

static void bench_generators() {
    constexpr size_t N = 20'000'000;

    {
        VZiggurat zig(4096);
        double acc = 0.0;
        perf_ctl().enable();
        const auto t0 = Clock::now();
        for (size_t i = 0; i < N; ++i) acc += zig();
        const double ms = ms_since(t0);
        perf_ctl().disable();
        g_sink += acc;
        std::printf("%-28s %8.3f ns/draw   (%.1f M draws/s)\n",
                    "ziggurat normal", ms * 1e6 / N, N / ms / 1e3);
    }
    {
        VUniformDistribution uni(4096);
        double acc = 0.0;
        perf_ctl().enable();
        const auto t0 = Clock::now();
        for (size_t i = 0; i < N; ++i) acc += uni();
        const double ms = ms_since(t0);
        perf_ctl().disable();
        g_sink += acc;
        std::printf("%-28s %8.3f ns/draw   (%.1f M draws/s)\n",
                    "xoshiro uniform double", ms * 1e6 / N, N / ms / 1e3);
    }
}

int main(int argc, char** argv) {
    constexpr size_t RUNS = 1000;
    constexpr size_t TOP_K = 10;

    const std::string which = (argc > 1) ? argv[1] : "all";
    // repeat multiplier: lengthens the measured region for stable perf
    // counters (hybrid CPUs multiplex events; aim for a few seconds)
    const size_t repeat = (argc > 2)
        ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10))
        : 1;
    const auto want = [&](const char* name) {
        return which == "all" || which == name;
    };

    if (want("mixed") || want("exact") || want("approx") ||
        want("nofb") || want("big")) {
        std::printf("--- end-to-end sample() ---\n");
    }

    // mirrors examples/benchmark.py: counts 1..50, forbidden 100..300
    if (want("mixed"))
        bench_sample("mixed (counts 1-50)",
                     make_workload(5000, 1, 50, RUNS, 100, 300, 1), TOP_K, repeat);

    if (want("exact"))
        bench_sample("all-exact (counts 1-20)",
                     make_workload(5000, 1, 20, RUNS, 100, 300, 2), TOP_K, repeat);

    if (want("approx"))
        bench_sample("mostly-approx (counts 150-200)",
                     make_workload(5000, 150, 200, RUNS, 100, 300, 3), TOP_K, repeat);

    if (want("nofb"))
        bench_sample("mixed, no forbidden",
                     make_workload(5000, 1, 50, RUNS, 0, 0, 4), TOP_K, repeat);

    if (want("big"))
        bench_sample("mixed, 100k items",
                     make_workload(100'000, 1, 50, 100, 2000, 6000, 5), 500, repeat);

    if (want("gen")) {
        std::printf("\n--- generator throughput ---\n");
        bench_generators();
    }

    std::printf("\n(sink=%g)\n", g_sink);
    return 0;
}