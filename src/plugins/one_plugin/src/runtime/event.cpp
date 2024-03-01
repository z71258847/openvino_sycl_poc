// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "event.hpp"
#include <list>
#include <vector>
#include <algorithm>

namespace ov {

void Event::wait() {
    if (_set)
        return;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    wait_impl();
    _set = true;
    return;
}

void Event::set() {
    if (_set)
        return;
    _set = true;
    set_impl();
}

bool Event::is_set() {
    if (_set)
        return true;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    _set = is_set_impl();
    return _set;
}

}  // namespace ov
