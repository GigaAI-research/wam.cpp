#pragma once

#include "wam/types.h"
#include "wam/version.h"

namespace wam {

struct Model;
struct Session;

Model * model_load(const ModelOptions & options);

Session * session_create(Model * model, const SessionOptions & options = {});

Prediction predict(Session * session, const Inputs & inputs);

Status session_reset(Session * session);

const ModelInfo & model_info(const Model * model);
void session_free(Session * session) noexcept;
void model_free(Model * model) noexcept;

} // namespace wam
