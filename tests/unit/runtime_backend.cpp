#include "support/test_utils.h"

#include "backends/ggml/backend_context.h"
#include "backends/ggml/debug_dump.h"
#include "backends/ggml/graph_context.h"
#include "backends/ggml/tensor_io.h"
#include "backends/ggml/weight_store.h"
#include "runtime/logger.h"
#include "runtime/telemetry.h"

#include "ggml-cpu.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace wam::internal;
    using wam::test::require;
    using wam::test::require_error;

    std::vector<std::string> messages;
    wam::RuntimeConfig runtime_config;
    runtime_config.log_level = wam::LogLevel::info;
    runtime_config.logger = [&](wam::LogLevel, std::string_view message) {
        messages.emplace_back(message);
    };
    runtime::Logger logger(runtime_config);
    logger.log(wam::LogLevel::debug, "hidden");
    logger.logf(wam::LogLevel::info, "loaded %d resources", 2);
    require(messages.size() == 1 && messages.front() == "loaded 2 resources",
            "logger filtering or callback dispatch is incorrect");

    wam::Telemetry telemetry;
    runtime::append_timing(telemetry, "ignored", 0.0);
    runtime::append_timing(telemetry, "compute", 1.25);
    require(telemetry.model_timings.size() == 1 &&
                telemetry.model_timings.front().name == "compute",
            "telemetry helper must keep positive timings only");

    const std::filesystem::path dump_root =
        std::filesystem::current_path() / "wam-phase4-debug-test";
    std::error_code filesystem_error;
    std::filesystem::remove_all(dump_root, filesystem_error);
    wam::DebugDumpConfig dump_config;
    dump_config.enabled = true;
    dump_config.directory = dump_root.string();
    ggml_backend::DebugDump dumper(dump_config);
    dumper.write("sample", std::vector<float>{1.0F, 2.0F}, {1, 2});
    require(std::filesystem::exists(dump_root / "sample.f32") &&
                std::filesystem::exists(dump_root / "sample.json"),
            "debug dumper did not create data and metadata files");
    std::filesystem::remove_all(dump_root, filesystem_error);

    require_error(
        [] { ggml_backend::WeightStore invalid(nullptr, 1); },
        wam::ErrorCode::invalid_argument,
        "WeightStore must reject a missing backend");
    require_error(
        [] { ggml_backend::GraphContext invalid(0); },
        wam::ErrorCode::invalid_argument,
        "GraphContext must reject an empty metadata arena");

    for (int iteration = 0; iteration < 32; ++iteration) {
        ggml_backend::BackendContext backend(ggml_backend_cpu_init());
        require(static_cast<bool>(backend), "CPU backend initialization failed");
        ggml_backend::BackendContext moved(std::move(backend));
        require(!backend && moved, "BackendContext move must transfer ownership");
        moved.reset(moved.get());
        require(static_cast<bool>(moved),
                "BackendContext reset with the owned handle must be stable");

        ggml_backend::WeightStore weights(moved.get(), 2);
        weights.define("weight", wam::DType::f32, {2});
        require_error(
            [&] { weights.define("invalid", wam::DType::f32, {0}); },
            wam::ErrorCode::failed_precondition,
            "WeightStore must reject non-positive tensor dimensions");
        require_error(
            [&] { weights.define("weight", wam::DType::f32, {2}); },
            wam::ErrorCode::failed_precondition,
            "WeightStore must reject duplicate tensors");
        weights.allocate();
        const std::vector<float> expected{3.0F, 4.0F};
        weights.set("weight", expected.data(), expected.size() * sizeof(float));
        require(ggml_backend::get_f32(weights.find("weight")) == expected,
                "WeightStore upload/readback failed");

        ggml_backend::GraphContext graph(1024 * 1024);
        ggml_tensor * input = ggml_new_tensor_1d(graph.get(), GGML_TYPE_F32, 2);
        ggml_tensor * output = ggml_dup(graph.get(), input);
        ggml_set_input(input);
        ggml_set_output(output);
        ggml_cgraph * computation = ggml_new_graph(graph.get());
        ggml_build_forward_expand(computation, output);
        graph.allocate(computation,
                       ggml_backend_get_default_buffer_type(moved.get()));
        require_error(
            [&] {
                graph.allocate(computation,
                               ggml_backend_get_default_buffer_type(moved.get()));
            },
            wam::ErrorCode::invalid_argument,
            "GraphContext must reject repeated allocation");
    }
    return 0;
}
