#pragma once

#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <iostream>

#include "Marsaglia.hpp"
#include "Uniform.hpp"
#include "MinHeap.hpp"
#include "VSet.hpp"

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
    Thompson(const std::vector<Item>& items)
        : set_(items.size())
    {
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
        set_.clear(forbidden);
        heap_.clear(num);
        refill();

        size_t i = 0;

        // exact sampling first
        for (i; i < border_; ++i) {
            const auto& item = precomp_[i];

            if (set_.contains(item.id)) continue;

            const double score = sample_beta(item);
            heap_.insert({item.id, float(score)});
        }

        // later approximation
        for (i; i < precomp_.size(); ++i) {
            const auto& item = precomp_[i];

            if (set_.contains(item.id)) continue;

            double z = normal();
            double score = item.mu + item.sigma * z;
            score = std::clamp(score, 0.0, 1.0);

            heap_.insert(IdWithScore{item.id, float(score)});
        }

        return heap_.getTopItems();
    }

private:
    struct alignas(64) Precomputed {
        double mu;      // 8 bytes
        double sigma;   // 8 bytes
        uint32_t id;         // 4 bytes
        double s_d;
        double s_c;
        double f_d;
        double f_c;
    };

    std::vector<Item> items_;
    std::vector<Precomputed> precomp_;

    VUniformDistribution udist_;
    VMarsaglia ndist_;

    FixedMinHeap<IdWithScore> heap_;
    VSet set_;
    size_t border_ = 0;

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

    void prepareDistribution() {
        if (items_.size() > 32) {
            udist_.resize(border_);
            ndist_.resize(items_.size());
        }
    }

    void refill() {
        if (items_.size() > 32) {
            udist_.refill();
            ndist_.refill();
        }
    }
};