#include <wam/wam.h>

int main() {
    wam::model_free(nullptr);
    wam::session_free(nullptr);
    return wam::Status::success() ? 0 : 1;
}
