#pragma once

#include <filesystem>

#include "optioryx/core/model.hpp"

namespace optioryx {

ProblemInstance load_instance(const std::filesystem::path& path);

}  // namespace optioryx
