#pragma once

#include "artifact.h"
#include "engine/engine.h"

#include "model_internal.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace wam::internal::gwp05 {

struct PreparedInputs {
    engine::EngineInputsView view;
    std::vector<engine::ImageView> images;
    std::vector<std::vector<std::uint8_t>> image_storage;
    std::vector<float> state;
    std::vector<float> noise;
    std::vector<float> reference_latent;
    std::vector<float> prompt_embedding;
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> attention_mask;
};

struct Runtime {
    std::mutex mutex;
    std::unique_ptr<engine::Engine> engine;
    std::uint64_t active_session = 0;
    std::uint64_t next_session = 1;
};

void prepare_vision(const Inputs & inputs, const ArtifactContract & artifact,
                    PreparedInputs & prepared);
void prepare_language(const Inputs & inputs, const ArtifactContract & artifact,
                      PreparedInputs & prepared);
PreparedInputs prepare_inputs(const Inputs & inputs, const ArtifactContract & artifact,
                              bool enable_prefix_cache, std::mt19937 & random);

void activate_session(Runtime & runtime, std::uint64_t session_id);
void reset_session_cache(Runtime & runtime, std::uint64_t session_id);

Prediction run_reference_prediction(Runtime & runtime, PreparedInputs & inputs,
                                    const ArtifactContract & artifact);

std::unique_ptr<ModelImpl> create_model(ModelInfo info, ArtifactContract artifact,
                                        const ModelOptions & options);

} // namespace wam::internal::gwp05
