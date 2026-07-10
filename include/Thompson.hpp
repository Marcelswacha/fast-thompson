#pragma once

#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>

#include "Marsaglia.hpp"
#include "Ziggurat.hpp"
#include "Uniform.hpp"
#include "MinHeap.hpp"

struct Item {
    Item(uint32_t id_, int s, int f)
    : id(id_), successes(s), failures(f) {}

    uint32_t id;
    int successes;
    int failures;

    bool isApprox(double skew_threshold = 0.1, int min_mass = 30) const {
        const double alpha = successes + 1.0;
        const double beta  = failures + 1.0;

        // ensure enough concentration (normal CLT region)
        if (alpha < min_mass || beta < min_mass)
            return false;

        const double sum = alpha + beta;

        const double skew =
            (2.0 * (beta - alpha) * std::sqrt(sum + 1.0)) /
            ((sum + 2.0) * std::sqrt(alpha * beta));

        return std::abs(skew) < skew_threshold;
    }
};

struct IdWithScore {
    uint32_t id;
    double score;

    bool operator<(const IdWithScore& other) const {
        return score < other.score;
    }
};

class Thompson {
public:
    explicit Thompson(const std::vector<Item>& items,
                      uint64_t seed = 0xdeadbeefcafebabeULL)
    {
        // Seed the two generators with *different* derived seeds. If they
        // shared a seed, the uniform stream used for gamma rejection would be
        // identical to the uniform stream feeding the normal generator,
        // producing correlated (invalid) samples.
        udist_.seed(seed);
        ndist_.seed(seed ^ 0x9e3779b97f4a7c15ULL);

        // Copy  items
        items_.reserve(items.size());
        for (const auto& item : items) {
            items_.push_back(item);
        }

        border_ = std::partition(items_.begin(), items_.end(),
                    [](const Item& i) {
                        return !i.isApprox();
                }) - items_.begin();

        prepareDistribution();
        buildIndex();

        precomp_.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const auto& item = items_[i];
            double alpha = item.successes + 1.0;
            double beta  = item.failures + 1.0;

            Precomputed p;
            p.id = item.id;
            p.s_d = (double)item.successes + 1.0 - 1.0 / 3.0;
            p.s_c= 1.0 / std::sqrt(9.0 * p.s_d);
            p.f_d = (double)item.failures + 1.0 - 1.0 / 3.0;
            p.f_c= 1.0 / std::sqrt(9.0 * p.f_d);

            if (i < border_) {
                p.mu = 0.0;
                p.sigma = 0.0;
            } else {
                double sum = alpha + beta;
                p.mu = alpha / sum;
                p.sigma = std::sqrt(alpha * beta / (sum * sum * (sum + 1)));
            }

            precomp_.push_back(p);
        }
    }

    std::vector<IdWithScore> sample(const size_t num, const std::vector<uint32_t>& forbidden) {
        std::memset(skip_.data(), 0, skip_.size());
        for (uint32_t id : forbidden) {
            const int64_t idx = findIndex(id);
            if (idx >= 0) skip_[idx] = 1;
        }

        heap_.clear(num);
        refill();  // batch-generate this call's randomness upfront

        size_t i = 0;

        // exact sampling first
        for (; i < border_; ++i) {
            if (skip_[i]) continue;

            const auto& item = precomp_[i];
            const double score = sample_beta(item);
            heap_.insert({item.id, score});
        }

        // later approximation
        for (; i < precomp_.size(); ++i) {
            if (skip_[i]) continue;

            const auto& item = precomp_[i];
            double z = normal();
            double score = item.mu + item.sigma * z;
            score = std::clamp(score, 0.0, 1.0);

            heap_.insert(IdWithScore{item.id, score});
        }

        return heap_.getTopItems();
    }

private:
    // One full cache line per item: the exact path touches s_d/s_c/f_d/f_c
    // per item, and 64-byte alignment guarantees exactly one line per access
    // (a packed 56-byte layout straddles lines and measures slower).
    struct alignas(64) Precomputed {
        double mu;
        double sigma;
        double s_d;
        double s_c;
        double f_d;
        double f_c;
        uint32_t id;
    };

    std::vector<Item> items_;
    std::vector<Precomputed> precomp_;

    VUniformDistribution udist_;
    VZiggurat ndist_;

    FixedMinHeap<IdWithScore> heap_;
    size_t border_ = 0;

    // Forbidden-id filtering: per-item lookup must be branch-predictable and
    // sequential, so we hash only the (few) forbidden ids into a dense
    // skip-byte per item position, instead of probing a hash set for every
    // item in the hot loop.
    std::vector<uint8_t> skip_;
    std::vector<uint32_t> mapKeys_;   // open addressing, key = id + 1, 0 = empty
    std::vector<uint32_t> mapVals_;   // value = position in precomp_

    inline double normal() {
        return ndist_();
    }

    inline double uniform() {
        return udist_();
    }

    double sample_gamma(double d, double c) {
        while (true) {
                double x = normal(); // standard normal
                double v = 1.0 + c * x;
                if (v <= 0) continue;
                v = v * v * v;
                double x2 = x * x;
                double x4 = x2 * x2;

                double u = uniform();
                if (u < 1 - 0.0331 * x4) return d * v;
                if (std::log(u) < 0.5 * x2 + d * (1 - v + std::log(v))) return d * v;
        }
    }

    double sample_beta(const Precomputed& p) {
        double g1 = sample_gamma(p.s_d, p.s_c);
        double g2 = sample_gamma(p.f_d, p.f_c);

        return g1 / (g1 + g2);
    }

    void buildIndex() {
        skip_.assign(items_.size(), 0);

        size_t cap = 32;
        while (cap < items_.size() * 2) cap <<= 1;   // load factor <= 0.5
        mapKeys_.assign(cap, 0);
        mapVals_.assign(cap, 0);

        const size_t mask = cap - 1;
        for (size_t pos = 0; pos < items_.size(); ++pos) {
            const uint32_t key = items_[pos].id + 1;
            size_t idx = hash(key) & mask;
            while (mapKeys_[idx] != 0) idx = (idx + 1) & mask;
            mapKeys_[idx] = key;
            mapVals_[idx] = static_cast<uint32_t>(pos);
        }
    }

    int64_t findIndex(uint32_t id) const {
        const uint32_t key = id + 1;
        const size_t mask = mapKeys_.size() - 1;
        size_t idx = hash(key) & mask;
        while (mapKeys_[idx] != 0) {
            if (mapKeys_[idx] == key) return mapVals_[idx];
            idx = (idx + 1) & mask;
        }
        return -1;
    }

    static inline uint32_t hash(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    void prepareDistribution() {
        // Batch strategy: generate one sample() call's worth of randomness
        // upfront in a single sequential, vectorized pass (cache friendly),
        // instead of lazily refilling in the middle of the sampling loop.
        //
        // Budget per call:
        //  - each exact item draws 2 gammas; each Marsaglia-Tsang gamma draw
        //    consumes >= 1 normal and >= 1 uniform, with ~96% acceptance and
        //    occasional squeeze-test retries -> budget ~2.25 of each
        //  - each approx item draws 1 normal
        //
        // If a call exceeds the budget (rejection streaks), the generators
        // still refill themselves lazily on exhaustion, so under-sizing is a
        // performance detail, never a correctness issue.
        const size_t exact = border_;
        const size_t approx = items_.size() - border_;

        udist_.resize(2 * exact + exact / 4 + 32);
        ndist_.resize(2 * exact + exact / 4 + approx + 32);
    }

    void refill() {
        udist_.refill();
        ndist_.refill();
    }
};