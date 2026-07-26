#include "debug.h"

#include <cstdlib>
#include <fstream>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace wam::internal::fastwam::debug {
namespace {

std::string directory() {
    const char * value = std::getenv("WAM_FASTWAM_DEBUG_DIR");
    return value && *value ? std::string(value) : std::string();
}

void create_directory(const std::string & path) {
#ifdef _WIN32
    (void) _mkdir(path.c_str());
#else
    (void) mkdir(path.c_str(), 0755);
#endif
}

template <typename T>
void write(const std::string & name, const char * extension,
           const std::vector<T> & values,
           const std::vector<std::int64_t> & shape) {
    const std::string root = directory();
    if (root.empty()) return;
    create_directory(root);
    const std::string separator =
        root.back() == '/' || root.back() == '\\' ? "" : "/";
    const std::string data_path = root + separator + name + extension;
    std::ofstream data(data_path, std::ios::binary | std::ios::trunc);
    if (!data) return;
    data.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    std::ofstream metadata(root + separator + name + ".shape", std::ios::trunc);
    if (!metadata) return;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index) metadata << ',';
        metadata << shape[index];
    }
    metadata << '\n';
}

} // namespace

bool enabled() { return !directory().empty(); }

void dump(const std::string & name, const std::vector<float> & values,
          const std::vector<std::int64_t> & shape) {
    write(name, ".f32", values, shape);
}

void dump(const std::string & name, const std::vector<ggml_bf16_t> & values,
          const std::vector<std::int64_t> & shape) {
    write(name, ".bf16", values, shape);
}

} // namespace wam::internal::fastwam::debug
