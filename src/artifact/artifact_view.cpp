#include "artifact/artifact_view.h"

#include <utility>

namespace wam::internal::artifact {

ArtifactView ArtifactView::open(const std::string & input_path) {
    ArtifactBundle bundle = resolve_bundle(input_path);
    std::shared_ptr<GgufReader> reader =
        GgufReader::open(bundle.model_path.string());
    return ArtifactView(std::move(bundle), std::move(reader));
}

ArtifactView::ArtifactView(ArtifactBundle bundle,
                           std::shared_ptr<GgufReader> reader)
    : bundle_(std::move(bundle)), reader_(std::move(reader)) {}

const ArtifactBundle & ArtifactView::bundle() const noexcept { return bundle_; }
const GgufReader & ArtifactView::gguf() const noexcept { return *reader_; }
std::shared_ptr<GgufReader> ArtifactView::shared_gguf() const noexcept {
    return reader_;
}
const std::string & ArtifactView::path() const noexcept { return reader_->path(); }
std::uint64_t ArtifactView::file_size() const noexcept {
    return reader_->file_size();
}
bool ArtifactView::has(const std::string & key) const noexcept {
    return reader_->has(key);
}
std::string ArtifactView::require_string(const std::string & key) const {
    return reader_->require_string(key);
}
std::uint32_t ArtifactView::require_u32(const std::string & key) const {
    return reader_->require_u32(key);
}
std::uint64_t ArtifactView::require_u64(const std::string & key) const {
    return reader_->require_u64(key);
}
float ArtifactView::require_f32(const std::string & key) const {
    return reader_->require_f32(key);
}
bool ArtifactView::require_bool(const std::string & key) const {
    return reader_->require_bool(key);
}
std::vector<float> ArtifactView::require_f32_array(
    const std::string & key) const {
    return reader_->require_f32_array(key);
}
std::vector<float> ArtifactView::optional_f32_array(
    const std::string & key) const {
    return reader_->optional_f32_array(key);
}
std::vector<std::string> ArtifactView::require_string_array(
    const std::string & key) const {
    return reader_->require_string_array(key);
}
std::vector<std::uint32_t> ArtifactView::require_u32_array(
    const std::string & key) const {
    return reader_->require_u32_array(key);
}
std::vector<std::int32_t> ArtifactView::require_i32_array(
    const std::string & key) const {
    return reader_->require_i32_array(key);
}
const GgufTensorInfo * ArtifactView::find_tensor(
    const std::string & name) const noexcept {
    return reader_->find_tensor(name);
}
const GgufTensorInfo & ArtifactView::require_tensor(
    const std::string & name) const {
    return reader_->require_tensor(name);
}
const std::vector<GgufTensorInfo> & ArtifactView::tensors() const noexcept {
    return reader_->tensors();
}
std::vector<float> ArtifactView::read_f32_tensor(
    const std::string & name) const {
    return reader_->read_f32_tensor(name);
}
void ArtifactView::require_shape(
    const std::string & name,
    const std::vector<std::int64_t> & shape) const {
    reader_->require_shape(name, shape);
}

} // namespace wam::internal::artifact
