#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/model.hpp"

namespace optioryx {

void write_solution_json(const std::filesystem::path& path, const ProblemInstance& instance, const Solution& solution);
void write_benchmark_runs_csv(const std::filesystem::path& path, const std::vector<BenchmarkRecord>& records);
void write_convergence_csv(const std::filesystem::path& path, const std::vector<ConvergenceRecord>& records);
void write_text_file(const std::filesystem::path& path, const std::string& content);

}  // namespace optioryx
