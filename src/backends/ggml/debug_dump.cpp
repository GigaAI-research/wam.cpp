#include "backends/ggml/debug_dump.h"

#include "ggml-backend.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace wam::internal::ggml_backend {

DebugDump::DebugDump(DebugDumpConfig config, ErrorReporter report_error)
    : config_(std::move(config)), report_error_(std::move(report_error)) {}

bool DebugDump::enabled() const noexcept {
    return config_.enabled && !config_.directory.empty();
}

bool DebugDump::audit_graph_dtypes() const noexcept {
    return config_.audit_graph_dtypes;
}

void DebugDump::report(std::string message) const {
    if (report_error_) report_error_(message);
}

template <typename T>
void DebugDump::write_binary(
    std::string_view name, std::string_view extension, std::string_view dtype,
    const std::vector<T> & values,
    const std::vector<std::int64_t> & shape) const {
    if (!enabled()) return;
    std::error_code error;
    std::filesystem::create_directories(config_.directory, error);
    if (error) {
        report("cannot create debug dump directory: " + error.message());
        return;
    }
    const std::filesystem::path root(config_.directory);
    const std::filesystem::path data_path =
        root / (std::string(name) + std::string(extension));
    std::ofstream data(data_path, std::ios::binary | std::ios::trunc);
    if (!data) {
        report("cannot write debug dump: " + data_path.string());
        return;
    }
    data.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!data) {
        report("cannot write debug dump: " + data_path.string());
        return;
    }
    const std::filesystem::path metadata_path =
        root / (std::string(name) + ".json");
    std::ofstream metadata(metadata_path, std::ios::trunc);
    if (!metadata) {
        report("cannot write debug metadata: " + metadata_path.string());
        return;
    }
    metadata << "{\"format\":\"wam-tensor-dump-v1\",\"dtype\":\""
             << dtype << "\",\"shape\":[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index) metadata << ',';
        metadata << shape[index];
    }
    metadata << "],\"elements\":" << values.size() << "}\n";
}

void DebugDump::write(std::string_view name, const std::vector<float> & values,
                      const std::vector<std::int64_t> & shape) const {
    write_binary(name, ".f32", "float32", values, shape);
}

void DebugDump::write(std::string_view name,
                      const std::vector<ggml_bf16_t> & values,
                      const std::vector<std::int64_t> & shape) const {
    write_binary(name, ".bf16", "bfloat16", values, shape);
}

void DebugDump::write_tensor(std::string_view name, ggml_tensor * tensor) const {
    if (!enabled() || !tensor || !tensor->buffer) return;
    std::vector<std::int64_t> shape;
    for (int dimension = ggml_n_dims(tensor) - 1; dimension >= 0; --dimension) {
        shape.push_back(tensor->ne[dimension]);
    }
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0,
                                values.size() * sizeof(float));
        write(name, values, shape);
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0,
                                values.size() * sizeof(ggml_bf16_t));
        write(name, values, shape);
    }
}

void DebugDump::audit_mixed_binary_nodes(
    std::string_view graph_name, ggml_cgraph * graph) const {
    if (!audit_graph_dtypes() || !graph) return;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        const ggml_tensor * node = ggml_graph_node(graph, index);
        if (!node->src[0] || !node->src[1] ||
            node->src[0]->type == node->src[1]->type) continue;
        if (node->op != GGML_OP_ADD && node->op != GGML_OP_SUB &&
            node->op != GGML_OP_MUL && node->op != GGML_OP_DIV) continue;
        std::ostringstream message;
        message << "dtype audit graph=" << graph_name << " node=" << index
                << " op=" << ggml_op_name(node->op)
                << " dst=" << ggml_type_name(node->type)
                << " src0=" << ggml_type_name(node->src[0]->type)
                << " src1=" << ggml_type_name(node->src[1]->type);
        report(message.str());
    }
}

} // namespace wam::internal::ggml_backend
