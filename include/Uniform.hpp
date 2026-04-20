#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "Xoshiro.hpp"

class VUniformDistribution {
    static constexpr size_t alignment = 32;

public:
    explicit VUniformDistribution(size_t capacity = 1024, double scale = 1.0, double offset = 0.0)
        : _samples(nullptr), _idx(0), _capacity(0), _rng(capacity / 2), _scale(scale), _offset(offset)
    {
        init(capacity);
        refill();
    }

    ~VUniformDistribution() {
        if (_samples) std::free(_samples);
    }

    VUniformDistribution(const VUniformDistribution&) = delete;
    VUniformDistribution& operator=(const VUniformDistribution&) = delete;

    VUniformDistribution(VUniformDistribution&&) = delete;
    VUniformDistribution& operator=(VUniformDistribution&&) = delete;

    double operator()() {
        if (_idx == _capacity) {
            refill();
            _idx = 0;
        }
        return _samples[_idx++];
    }

    void refill() {
        _rng.refill();

        constexpr double INV32 = 1.0 / 4294967296.0;
        const double mul = INV32 * _scale;

        size_t j = 0;

        for (size_t i = 0; i < _capacity / 2; ++i) {
            uint64_t x = _rng();

            uint32_t lo = static_cast<uint32_t>(x);
            uint32_t hi = static_cast<uint32_t>(x >> 32);

            _samples[j++] = (static_cast<double>(lo) * mul) + _offset;
            _samples[j++] = (static_cast<double>(hi) * mul) + _offset;
        }

        _idx = 0;
    }

    void resize(size_t capacity) {
        capacity = roundCapacity(capacity);

        double* oldSamples = _samples;

        allocate(capacity);

        _rng.resize(capacity / 2);

        _idx = 0;

        if (oldSamples) std::free(oldSamples);

        refill();
    }

    size_t size() const { return _capacity; }

private:
    double* _samples;
    VXoshiro _rng;

    size_t _idx;
    size_t _capacity;
    double _scale;
    double _offset;

    void init(size_t capacity) {
        capacity = roundCapacity(capacity);
        allocate(capacity);

        _rng.resize(capacity / 2);

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
        cap = (cap + 31) & ~static_cast<size_t>(31);
        return cap == 0 ? 32 : cap;
    }
};