#pragma once

#include <cstdint>
#include <cstring>

class VSet {
    static constexpr uint32_t empty = 0u;
    static constexpr double loadFactor = 0.5;

public:
    explicit VSet(size_t capacity = 256)
        : _data(nullptr), _capacity(0), _size(0)
    {
        init(capacity);
    }

    ~VSet() {
        if (_data) std::free(_data);
    }

    VSet(const VSet&) = delete;
    VSet& operator=(const VSet&) = delete;

    void insert(uint32_t element) {
        if (needsResize()) {
            resize(_capacity * 2);
        }

        size_t idx = hash(element) % _capacity;

        while (_data[idx] != empty) {
            if (_data[idx] == element) return;
            idx = (idx + 1) % _capacity;
        }

        _data[idx] = element;
        ++_size;
    }

    bool contains(uint32_t element) const {
        if (_capacity == 0) return false;

        size_t idx = hash(element) % _capacity;

        while (_data[idx] != empty) {
            if (_data[idx] == element) return true;
            idx = (idx + 1) % _capacity;
        }

        return false;
    }

    void clear() {
        std::memset(_data, empty, _capacity * sizeof(uint32_t));
        _size = 0;
    }

    void clear(const std::vector<uint32_t>& elements) {
        std::memset(_data, empty, _capacity * sizeof(uint32_t));
        _size = 0;

        size_t count = std::min(elements.size(), static_cast<size_t>(_capacity));
        std::memcpy(_data, elements.data(), count * sizeof(uint32_t));
        _size = count;
    }

    size_t size() const { return _size; }

private:
    uint32_t* _data;
    size_t _capacity;
    size_t _size;

    void init(size_t capacity) {
        capacity = roundCapacity(capacity);
        allocate(capacity);

        std::memset(_data, empty, _capacity * sizeof(uint32_t));
    }

    void resize(size_t newCapacity) {
        newCapacity = roundCapacity(newCapacity);

        uint32_t* oldData = _data;
        size_t oldCap = _capacity;

        allocate(newCapacity);

        std::memset(_data, 0, _capacity * sizeof(uint32_t));

        _size = 0;

        // rehash
        for (size_t i = 0; i < oldCap; ++i) {
            if (oldData[i] != empty) {
                insert(oldData[i]);
            }
        }

        std::free(oldData);
    }

    void allocate(size_t capacity) {
        _data = static_cast<uint32_t*>(
            std::aligned_alloc(32, capacity * sizeof(uint32_t))
        );

        if (!_data) {
            throw std::bad_alloc();
        }

        _capacity = capacity;
    }

    bool needsResize() const {
        return _capacity > 0 &&
               (static_cast<double>(_size + 1) / _capacity > loadFactor);
    }

    static size_t roundCapacity(size_t cap) {
        cap = (cap + 31) & ~static_cast<size_t>(31);
        return cap == 0 ? 32 : cap;
    }

    static inline uint32_t hash(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }
};