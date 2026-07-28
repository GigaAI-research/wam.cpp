#pragma once

#include "wam/runtime_config.h"

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace wam::internal::ggml_backend {

class DebugDump final {
public:
    using ErrorReporter = std::function<void(std::string_view)>;

    explicit DebugDump(DebugDumpConfig config = {},
                       ErrorReporter report_error = {});

    bool enabled() const noexcept;
    bool audit_graph_dtypes() const noexcept;

    void write(std::string_view name, const std::vector<float> & values,
               const std::vector<std::int64_t> & shape) const;
    void write(std::string_view name, const std::vector<ggml_bf16_t> & values,
               const std::vector<std::int64_t> & shape) const;
    void write_tensor(std::string_view name, ggml_tensor * tensor) const;
    void audit_mixed_binary_nodes(std::string_view graph_name,
                                  ggml_cgraph * graph) const;

private:
    template <typename T>
    void write_binary(std::string_view name, std::string_view extension,
                      std::string_view dtype, const std::vector<T> & values,
                      const std::vector<std::int64_t> & shape) const;
    void report(std::string message) const;

    DebugDumpConfig config_;
    ErrorReporter report_error_;
};

} // namespace wam::internal::ggml_backend
