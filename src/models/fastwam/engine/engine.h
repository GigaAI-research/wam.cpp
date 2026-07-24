#pragma once

#include <memory>

namespace wam::internal::fastwam {

struct ArtifactContract;
struct PreparedInputs;
struct CoreAction;

namespace engine {

class Engine;
struct EngineOptions;

std::unique_ptr<Engine> create_engine(const ArtifactContract & artifact,
                                      const EngineOptions & options);
CoreAction predict(Engine & engine, const PreparedInputs & inputs);
void reset(Engine & engine);

} // namespace engine
} // namespace wam::internal::fastwam
