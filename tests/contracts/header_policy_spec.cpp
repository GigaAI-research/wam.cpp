#include "wam/policy_spec.h"

int main() {
    return wam::kPolicySpecSchemaVersion == 3 ? 0 : 1;
}
