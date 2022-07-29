// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <list>
#include <unordered_map>
#include <functional>
#include <iostream>

namespace cldnn {

struct primitive_impl;

/// @brief LRU cache which remove the least recently used data when cache is full.
template<typename TypeK, typename TypeD>
class LRUCache {
public:
    struct CacheEntry {
        TypeD   data;
        size_t  size;   // the size of data. it is necessary to check total data size is limited to capacity
    };

private:
    std::list<TypeK> order;
    std::unordered_map<TypeK, std::pair<typename std::list<TypeK>::iterator, LRUCache::CacheEntry>> cache_entry_map;
    size_t curr_data_size;
    const size_t capacity;

public:
    explicit LRUCache(size_t caps) : curr_data_size(0), capacity(caps) {}

    ~LRUCache() {
        clear();
    }

    TypeD get_lru_element() {
        if (order.size()) {
            return cache_entry_map[order.back()].second.data;
        } else {
            return nullptr;
        }
    }

    TypeD add(TypeK key, std::function<LRUCache::CacheEntry(void)> create_new_data, bool* last_element_popped = nullptr) {
        if (!create_new_data)
            throw std::runtime_error("Please add function to create new data");

        TypeD data;
        if (cache_entry_map.find(key) == cache_entry_map.end()) {
            auto new_entry = create_new_data();
            add_new_entry(key, new_entry, last_element_popped);
            data = cache_entry_map[key].second.data;
        } else {
            throw std::runtime_error("Already have entry with same key");
        }
        return data;
    }

    bool has(TypeK key) const {
        return (cache_entry_map.find(key) != cache_entry_map.end());
    }

    TypeD get(TypeK key) {
        TypeD data;
        if (cache_entry_map.find(key) != cache_entry_map.end()) {
            // Move current data to front of deque
            order.erase(cache_entry_map[key].first);
            order.push_front(key);
            cache_entry_map[key].first = order.begin();
            data = cache_entry_map[key].second.data;
        } else {
            throw std::runtime_error("Fail to get entry");
        }

        return data;
    }

    void clear() {
        order.clear();
        cache_entry_map.clear();
        curr_data_size = 0;
    }

    size_t count() const {
       return cache_entry_map.size();
    }

    size_t get_current_data_size() {
        return curr_data_size;
    }

    std::list<TypeK> get_all_keys() const {
        return order;
    }

private:
    void add_new_entry(TypeK key, LRUCache::CacheEntry entry, bool* popped_last_element) {
        if (popped_last_element)
            *popped_last_element = false;
        if (cache_entry_map.find(key) == cache_entry_map.end()) {
            if (capacity != 0 && curr_data_size > 0 && capacity < (curr_data_size + entry.size)) {
                //  Remove cache at the end of order
                curr_data_size -= cache_entry_map[order.back()].second.size;
                cache_entry_map.erase(order.back());
                order.pop_back();
                if (popped_last_element)
                    *popped_last_element = true;
            }
        } else {
            order.erase(cache_entry_map[key].first);
        }
        order.push_front(key);
        cache_entry_map[key] = std::make_pair(order.begin(), entry);
        curr_data_size += entry.size;
    }
};

using ImplementationsCache = LRUCache<std::string, std::shared_ptr<primitive_impl>>;
}  // namespace cldnn