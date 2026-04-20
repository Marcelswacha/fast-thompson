#pragma once

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

class VXoshiro {
    static constexpr size_t alignment = 32;

public:
    explicit VXoshiro(size_t capacity = 1024, uint64_t seed = 0xdeadbeefcafebabeULL)
        : _buffer(nullptr), _idx(0), _capacity(0)
    {
        init(capacity);
        seed_all(seed);
    }

    ~VXoshiro() {
        if (_buffer) std::free(_buffer);
    }

    VXoshiro(const VXoshiro&) = delete;
    VXoshiro& operator=(const VXoshiro&) = delete;

    VXoshiro(VXoshiro&&) = delete;
    VXoshiro& operator=(VXoshiro&&) = delete;

    uint64_t operator()() {
        if (_idx == _capacity) {
            _idx = 0;
        }
        return _buffer[_idx++];
    }

    void seed(uint64_t seed_val) {
        seed_all(seed_val);
    }

    void refill() {
        for (size_t i = 0; i < _capacity; i += 4) {
            __m256i rnd = next_vector();
            _mm256_store_si256(reinterpret_cast<__m256i*>(&_buffer[i]), rnd);
        }
    }

    void resize(size_t capacity) {
        capacity = roundCapacity(capacity);

        uint64_t* oldBuffer = _buffer;

        allocate(capacity);
        _idx = 0;

        if (oldBuffer) std::free(oldBuffer);
    }


    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

private:
    __m256i _s0, _s1, _s2, _s3;

    uint64_t* _buffer;
    size_t _idx;
    size_t _capacity;

    void init(size_t capacity) {
        capacity = roundCapacity(capacity);
        allocate(capacity);

        std::memset(_buffer, 0, _capacity * sizeof(uint64_t));
    }

    void allocate(size_t capacity) {
        _buffer = static_cast<uint64_t*>(
            std::aligned_alloc(alignment, capacity * sizeof(uint64_t))
        );

        if (!_buffer) {
            throw std::bad_alloc();
        }

        _capacity = capacity;
    }

    void seed_all(uint64_t seed) {
        uint64_t x = seed;
        uint64_t tmp[4];

        for (int i = 0; i < 4; ++i) {
            tmp[i] = splitmix64(x);
        }

        _s0 = _mm256_set1_epi64x(tmp[0]);
        _s1 = _mm256_set1_epi64x(tmp[1]);
        _s2 = _mm256_set1_epi64x(tmp[2]);
        _s3 = _mm256_set1_epi64x(tmp[3]);
    }

    __m256i next_vector() {
        __m256i result =
            _mm256_add_epi64(rotl(_mm256_add_epi64(_s0, _s3), 23), _s0);

        __m256i t = _mm256_slli_epi64(_s1, 17);

        _s2 = _mm256_xor_si256(_s2, _s0);
        _s3 = _mm256_xor_si256(_s3, _s1);
        _s1 = _mm256_xor_si256(_s1, _s2);
        _s0 = _mm256_xor_si256(_s0, _s3);

        _s2 = _mm256_xor_si256(_s2, t);
        _s3 = rotl(_s3, 45);

        return result;
    }

    static inline __m256i rotl(__m256i v, int k) {
        return _mm256_or_si256(
            _mm256_slli_epi64(v, k),
            _mm256_srli_epi64(v, 64 - k)
        );
    }

    static size_t roundCapacity(size_t cap) {
        cap = (cap + 31) & ~static_cast<size_t>(31);
        return cap == 0 ? 32 : cap;
    }

    static inline uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};