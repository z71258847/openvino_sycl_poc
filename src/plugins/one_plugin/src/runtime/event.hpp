// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <memory>

namespace ov {

struct Event {
public:
    using Ptr = std::shared_ptr<Event>;
    Event() = default;

    void wait();
    void set();
    bool is_set();
    virtual void reset() {
        _set = false;
    }


protected:
    bool _set = false;

    virtual void wait_impl() = 0;
    virtual void set_impl() = 0;
    virtual bool is_set_impl() = 0;
};

}  // namespace ov
