#pragma once

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

#include "Uniform.hpp"

class VMarsaglia {
    static constexpr size_t alignment = 32;

public:
    explicit VMarsaglia(size_t capacity = 1024)
        : _samples(nullptr),
          _idx(0),
          _capacity(0),
          _uniform(capacity, -2.0, 1.0)
    {
        init(capacity);
        refill();
    }

    ~VMarsaglia() {
        if (_samples) std::free(_samples);
    }

    VMarsaglia(const VMarsaglia&) = delete;
    VMarsaglia& operator=(const VMarsaglia&) = delete;
    VMarsaglia(VMarsaglia&&) = delete;
    VMarsaglia& operator=(VMarsaglia&&) = delete;

    double operator()() {
        if (_idx == _capacity) {
            refill();
            _idx = 0;
        }

        return _samples[_idx++];
    }

    void seed(uint64_t seed_val) {
        _uniform.seed(seed_val);
        refill();
    }

    void refill() {
        _uniform.refill();

        size_t j = 0;

        while (j < _capacity) {
            double u, v, s;
            do {
                u = _uniform();
                v = _uniform();
                s = u * u + v * v;
            } while (s >= 1.0 || s == 0.0);

            const double mul = std::sqrt(-2.0 * std::log(s) / s);

            _samples[j++] = u * mul;
            _samples[j++] = v * mul;
        }

        _idx = 0;
    }

    void resize(size_t capacity) {
        capacity = roundCapacity(capacity);

        double* old = _samples;

        allocate(capacity);

        _uniform.resize(capacity);

        _idx = 0;

        if (old) std::free(old);

        refill();
    }

    size_t size() const { return _capacity; }

private:
    double* _samples;
    VUniformDistribution _uniform;

    size_t _idx;
    size_t _capacity;

    void init(size_t capacity) {
        capacity = roundCapacity(capacity);
        allocate(capacity);

        _uniform.resize(capacity);

        std::memset(_samples, 0, _capacity * sizeof(double));
    }

    void allocate(size_t capacity) {
        _samples = static_cast<double*>(
            std::aligned_alloc(alignment, capacity * sizeof(double))
        );

        if (!_samples) {
            throw std::bad_alloc();
        }

        _capacity = capacity;
    }

    static size_t roundCapacity(size_t cap) {
        cap = (cap + 31) & ~size_t(31);
        return cap == 0 ? 32 : cap;
    }
};