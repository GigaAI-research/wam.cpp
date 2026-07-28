#pragma once

#include "wam/model.h"

#include <memory>

namespace wam::test {

struct FakeModelCounters {
    int models_destroyed = 0;
    int sessions_created = 0;
    int sessions_destroyed = 0;
    int predictions = 0;
};

Model make_fake_model(const std::shared_ptr<FakeModelCounters> & counters,
                      bool return_null_session = false);

} // namespace wam::test
