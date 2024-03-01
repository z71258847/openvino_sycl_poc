// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"

#include <memory>

namespace ov {

class Engine {
public:
    /// Default destructor
    virtual ~Engine() = default;

    /// Factory method which creates engine object with impl configured by @p engine_type
    /// @param engine_type requested engine type
    /// @param runtime_type requested execution runtime for the engine. @note some runtime/engine types configurations might be unsupported
    /// @param device specifies the device which the engine is created for
    /// @param configuration options for the engine
    static std::shared_ptr<ov::Engine> create(/* engine_types engine_type, runtime_types runtime_type ,*/ const Device::Ptr device);

    /// Factory method which creates engine object with impl configured by @p engine_type
    /// @param engine_type requested engine type
    /// @param runtime_type requested execution runtime for the engine. @note some runtime/engine types configurations might be unsupported
    /// @param configuration options for the engine
    /// @note engine is created for the first device returned by devices query
    static std::shared_ptr<ov::Engine> create(/* engine_types engine_type, runtime_types runtime_type */);

protected:
    explicit Engine(const Device::Ptr device);
    const Device::Ptr _device;
};

}  // namespace ov
