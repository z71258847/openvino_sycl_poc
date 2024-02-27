// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(infer_request, task_executor, callback_executor)
    , m_infer_request(infer_request)
    , m_wait_executor(wait_executor) {
    if (infer_request->use_external_queue()) {
        m_pipeline.clear();
        m_pipeline.emplace_back(wait_executor,
                        [this] {
                        });
    }
}
void AsyncInferRequest::start_async() {
    Parent::start_async();
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_gpu
}  // namespace ov
