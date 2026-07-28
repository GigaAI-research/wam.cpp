#pragma once

#include "wam/policy_spec.h"
#include "wam/observation.h"
#include "wam/runtime_config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam {

class Session;

namespace internal {
class ModelImpl;
struct ModelAccess;
} // namespace internal

struct Capabilities {
    bool action = false;
    bool auxiliary_outputs = false;
    bool raw_images = false;
    bool token_input = false;
    bool precomputed_embedding = false;
    bool explicit_action_noise = false;
    bool batch_inference = false;
    bool concurrent_sessions = false;
    bool arbitrary_token_input = false;
    bool fixed_token_input = false;
    std::vector<Backend> backends;
    std::vector<ComputePrecision> compute_precisions;
};

struct RuntimeComponentInfo {
    std::string name;
    bool loaded = false;
    std::uint64_t device_bytes = 0;
    double load_milliseconds = 0.0;
    double unload_milliseconds = 0.0;
};

struct ArtifactComponentInfo {
    std::string name;
    DType source_dtype = DType::unknown;
    DType stored_dtype = DType::unknown;
    std::uint64_t tensor_count = 0;
    std::uint64_t elements = 0;
    std::uint64_t source_bytes = 0;
    std::uint64_t stored_bytes = 0;
};

struct ModelInfo {
    std::string architecture;
    std::string artifact_path;
    std::uint64_t artifact_bytes = 0;
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::unknown;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;
    Capabilities capabilities;
    std::shared_ptr<const PolicySpec> policy_spec;
    std::vector<ArtifactComponentInfo> artifact_components;
    std::vector<RuntimeComponentInfo> runtime_components;
};

class Model final {
public:
    static Model load(const std::string & artifact_path,
                      const RuntimeConfig & config = {});

    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Session create_session(const SessionConfig & config = {}) const;
    const ModelInfo & info() const;

private:
    explicit Model(std::shared_ptr<internal::ModelImpl> impl);

    std::shared_ptr<internal::ModelImpl> impl_;
    friend struct internal::ModelAccess;
};

} // namespace wam
