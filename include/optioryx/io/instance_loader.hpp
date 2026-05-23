#pragma once

#include <filesystem>

#include "optioryx/core/model.hpp"

namespace optioryx {

ProblemInstance load_instance(const std::filesystem::path& path);
ProblemInstance load_fixture_instance(const std::filesystem::path& path);
ProblemInstance load_competition_instance(const std::filesystem::path& path);

}  // namespace optioryx
