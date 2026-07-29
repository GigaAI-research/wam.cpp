#include "wam/wam.h"

// Application-owned storage referenced by Observation must outlive predict().
wam::Prediction predict_once(const std::string & artifact,
                             const wam::Observation & observation) {
    wam::Pipeline pipeline = wam::Pipeline::load(artifact);
    return pipeline.predict(observation);
}
