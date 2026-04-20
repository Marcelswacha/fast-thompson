#pragma once

#include <cstdlib>
#include <new>
#include <vector>

template <typename T>
class FixedMinHeap {
public:
    explicit FixedMinHeap(size_t capacity = 16)
        : data_(nullptr), size_(0), capacity_(0)
    {
        allocate(capacity);
    }

    ~FixedMinHeap() {
        std::free(data_);
    }

    FixedMinHeap(const FixedMinHeap&) = delete;
    FixedMinHeap& operator=(const FixedMinHeap&) = delete;

    // Reset heap, optionally change capacity
    void clear(size_t newCapacity = 0) {
        if (newCapacity > capacity_) {
            std::free(data_);
            allocate(newCapacity);
        }

        size_ = 0;
    }

    void insert(const T& item) {
        if (size_ < capacity_) {
            data_[size_] = item;
            heapifyUp(size_);
            ++size_;
        } else if (data_[0] < item) {
            data_[0] = item;
            heapifyDown(0);
        }
    }

    std::vector<T> getTopItems() const {
        return std::vector<T>(data_, data_ + size_);
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    T* data_;
    size_t size_;
    size_t capacity_;

    void allocate(size_t cap) {
        data_ = new T[cap];
        capacity_ = cap;
    }

    void swap(size_t a, size_t b) {
        T tmp = std::move(data_[a]);
        data_[a] = std::move(data_[b]);
        data_[b] = std::move(tmp);
    }

    void heapifyUp(size_t i) {
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (!(data_[i] < data_[parent])) break;
            swap(i, parent);
            i = parent;
        }
    }

    void heapifyDown(size_t i) {
        while (true) {
            size_t left  = 2 * i + 1;
            size_t right = 2 * i + 2;
            size_t smallest = i;

            if (left < size_ && data_[left] < data_[smallest])
                smallest = left;
            if (right < size_ && data_[right] < data_[smallest])
                smallest = right;

            if (smallest == i) break;

            swap(i, smallest);
            i = smallest;
        }
    }
};