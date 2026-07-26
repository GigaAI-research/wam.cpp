#pragma once

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

namespace wam::test {

class TempFile final {
public:
    explicit TempFile(const std::string & stem) {
        static std::atomic<unsigned long> sequence{0};
        path_ = std::filesystem::temp_directory_path() /
            ("wam-" + stem + "-" + std::to_string(getpid()) + "-" +
             std::to_string(sequence.fetch_add(1)) + ".gguf");
    }

    ~TempFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TempFile(const TempFile &) = delete;
    TempFile & operator=(const TempFile &) = delete;

    std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

} // namespace wam::test
