// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/implementation_args.hpp"
#include "extension/node_extension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/op_types.hpp"
#include "runtime/memory.hpp"
#include "transformations/utils/utils.hpp"

#include "sync_infer_request.hpp"
#include "remote_context.hpp"
#include "remote_tensor.hpp"
#include "compiled_model.hpp"
#include "variable_state.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <fstream>
#include <map>
#include <functional>
#include <utility>

namespace ov {
namespace intel_gpu {

// ----------------------------------------------------------------------------------------------- //
// ---------------------------- OpenVINO API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------------- //

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
}

void SyncInferRequest::infer() {
    auto model = std::dynamic_pointer_cast<const ov::intel_gpu::CompiledModel>(get_compiled_model())->get_model();


    std::cerr << "Execute model " << std::endl;

    std::map<std::shared_ptr<Node>, MemoryArgs> all_args;

    // dummy memory alloc
    for (auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        auto args_desc = node->get_default_descriptors();

        MemoryArgs args;
        if (ov::is_type<ov::op::v0::Parameter>(op)) {
            auto in = get_tensor(op->output(0));
            args[Argument::output(0)] = std::make_shared<Memory>(nullptr, args_desc.at(Argument::output(0)), in->data());
            all_args[op] = args;
            continue;
        }

        std::cerr << "Handle: " << op->get_friendly_name() << std::endl;
        for (const auto& kv : args_desc) {
            const auto& arg = kv.first;
            const auto& desc = kv.second;
            std::cerr << "\t" << arg << std::endl;
            if (arg.type() == ArgumentType::OUTPUT) {
                args[arg] = std::make_shared<Memory>(nullptr, desc);
                std::cerr << "\t\t" << "ALLOC: " << desc.to_string() << " ptr=" << args[arg]->ptr << std::endl;
            } else if (arg.type() == ArgumentType::INPUT) {
                std::cerr << "\t\t" << "USE PREV: "  << desc.to_string() << std::endl;
                auto in_op = op->get_input_node_shared_ptr(arg.id());
                args[arg] = all_args[in_op].at(Argument::output(op->get_input_source_output(arg.id()).get_index()));
            } else if (arg.type() == ArgumentType::WEIGHTS) {
                std::cerr << "\t\t" << "ALLOC WEIGHTS: " << desc.to_string() << std::endl;
                auto in_op = op->get_input_node_shared_ptr(1);
                void* ptr = const_cast<void*>(std::dynamic_pointer_cast<ov::op::v0::Constant>(in_op)->get_data_ptr());
                args[arg] = std::make_shared<Memory>(nullptr, desc, ptr);
            }
        }

        if (ov::is_type<ov::op::v0::Result>(op)) {
            auto desc = args_desc.at(Argument::input(0));
            auto out = make_tensor(desc.m_data_type, desc.m_shape.to_shape());
            set_tensor(op->output(0), { out, nullptr});

            auto in_op = op->get_input_node_shared_ptr(0);
            auto actual_out = Argument::output(op->get_input_source_output(0).get_index());
            std::cerr << "RESULT PTR: " << out->data() << std::endl;
            std::cerr << "Update " << in_op->get_friendly_name() << " " << actual_out  << std::endl;
            all_args[in_op].at(actual_out) = std::make_shared<Memory>(nullptr, args_desc.at(Argument::output(0)), out->data());
        }


        all_args[op] = args;
    }

    size_t i = 0;
    for (auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        auto executor = node->get_executor();
        Stream s;
        const auto& args = all_args[op];
        std::cerr << "Execute " << op->get_friendly_name() << std::endl;


        auto save_tensor = [](const Memory& m, std::string file_name) {
            std::ofstream os(file_name);
            os << m.shape() << std::endl;
            for (size_t i = 0; i < m.count(); i++) {
                os << static_cast<float*>(m.ptr)[i] << std::endl;
            }
        };
        for (auto& kv : args) {
            std::cerr << "\t" << kv.first << " ptr= " << kv.second->ptr << std::endl;
            std::stringstream file_name;
            file_name << "dump/" << i  << "_" << op->get_friendly_name() << "_"  << kv.first.type() << ".txt";
            if (kv.first.type() == ArgumentType::INPUT || kv.first.type() == ArgumentType::WEIGHTS)
                save_tensor(*kv.second, file_name.str());
        }
        executor->execute(s, args, {});
        for (auto& kv : args) {
            std::cerr << "\t" << kv.first << " ptr= " << kv.second->ptr << std::endl;
            std::stringstream file_name;
            file_name << "dump/" << i  << "_" << op->get_friendly_name() << "_"  << kv.first.type() << ".txt";
            if (kv.first.type() == ArgumentType::OUTPUT)
                save_tensor(*kv.second, file_name.str());
        }
        i++;
    }
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> ret{};
    for (const auto& pair : m_variables) {
        ret.emplace_back(pair.second, nullptr);
    }
    return ret;
}

// void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
// }

// void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
//     OPENVINO_THROW("[GPU] Cannot find input tensors for port ", port);
// }

// ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
//     return {};
// }

void SyncInferRequest::check_tensors() const {
}

}  // namespace intel_gpu
}  // namespace ov
