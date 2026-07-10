#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>

#include "Xoshiro.hpp"

// Buffered standard-normal generator using the ziggurat method
// (Marsaglia & Tsang 2000, Doornik's ZIGNOR variant, 128 layers).
//
// ~97.5% of draws take the fast path: one uint64 from the RNG, one multiply,
// one compare -- no sqrt/log/exp. The remaining ~2.5% (wedges and the tail)
// fall back to exp/log. This replaces the polar method, which paid a scalar
// sqrt + log per accepted *pair* plus a 21.5% rejection loop.
class VZiggurat {
    static constexpr size_t alignment = 32;
    static constexpr int    C = 128;                      // number of layers
    static constexpr double R = 3.442619855899;           // right-most x
    static constexpr double V = 9.91256303526217e-3;      // area per layer

public:
    explicit VZiggurat(size_t capacity = 1024)
        : _samples(nullptr), _idx(0), _capacity(0), _rng(1024)
    {
        initTables();
        init(capacity);
        refill();
    }

    ~VZiggurat() {
        if (_samples) std::free(_samples);
    }

    VZiggurat(const VZiggurat&) = delete;
    VZiggurat& operator=(const VZiggurat&) = delete;
    VZiggurat(VZiggurat&&) = delete;
    VZiggurat& operator=(VZiggurat&&) = delete;

    double operator()() {
        if (_idx == _capacity) {
            refill();
        }
        return _samples[_idx++];
    }

    void seed(uint64_t seed_val) {
        _rng.seed(seed_val);
        refill();
    }

    void refill() {
        for (size_t j = 0; j < _capacity; ++j) {
            _samples[j] = draw();
        }
        _idx = 0;
    }

    void resize(size_t capacity) {
        capacity = roundCapacity(capacity);

        double* old = _samples;
        allocate(capacity);
        _rng.resize(capacity);
        _idx = 0;

        if (old) std::free(old);

        refill();
    }

    size_t size() const { return _capacity; }

private:
    double _X[C + 1];   // layer right edges, X[0] = base, X[C] = 0
    double _F[C + 1];   // f(X[i]) = exp(-X[i]^2 / 2)

    double* _samples;
    size_t _idx;
    size_t _capacity;

    VXoshiro _rng;

    void initTables() {
        double f = std::exp(-0.5 * R * R);
        _X[0] = V / f;   // base layer extends past R to keep equal areas
        _X[1] = R;
        for (int i = 2; i < C; ++i) {
            _X[i] = std::sqrt(-2.0 * std::log(
                V / _X[i - 1] + std::exp(-0.5 * _X[i - 1] * _X[i - 1])));
        }
        _X[C] = 0.0;
        for (int i = 0; i <= C; ++i) {
            _F[i] = std::exp(-0.5 * _X[i] * _X[i]);
        }
    }

    inline double uniform01() {
        // (0, 1]: +1 keeps log() finite in the tail sampler
        return (static_cast<double>(_rng() >> 11) + 1.0) * (1.0 / 9007199254740992.0);
    }

    inline double draw() {
        for (;;) {
            const uint64_t bits = _rng();

            // bits 0..6: layer index; bits 11..63: signed 53-bit uniform
            const int i = static_cast<int>(bits & 0x7F);
            const double u =
                static_cast<double>(static_cast<int64_t>(bits) >> 11) *
                (1.0 / 4503599627370496.0);   // in [-1, 1)

            const double x = u * _X[i];

            // fast path: strictly inside the next (narrower) layer
            if (std::fabs(x) < _X[i + 1]) {
                return x;
            }

            if (i == 0) {
                // tail beyond R (Marsaglia's exponential method)
                double xx, yy;
                do {
                    xx = -std::log(uniform01()) / R;
                    yy = -std::log(uniform01());
                } while (yy + yy < xx * xx);
                return (u < 0.0) ? -(R + xx) : (R + xx);
            }

            // wedge: accept against the density
            const double y = _F[i + 1] + uniform01() * (_F[i] - _F[i + 1]);
            if (y < std::exp(-0.5 * x * x)) {
                return x;
            }
        }
    }

    void init(size_t capacity) {
        capacity = roundCapacity(capacity);
        allocate(capacity);
        _rng.resize(capacity);
        std::memset(_samples, 0, _capacity * sizeof(double));
    }

    void allocate(size_t capacity) {
        _samples = static_cast<double*>(
            std::aligned_alloc(alignment, capacity * sizeof(double)));

        if (!_samples) {
            throw std::bad_alloc();
        }

        _capacity = capacity;
    }

    static size_t roundCapacity(size_t cap) {
        cap = (cap + 31) & ~static_cast<size_t>(31);
        return cap == 0 ? 32 : cap;
    }
};