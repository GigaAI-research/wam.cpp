#include "wam/prediction.h"

int main() {
    return wam::Prediction{}.action.empty() ? 0 : 1;
}
