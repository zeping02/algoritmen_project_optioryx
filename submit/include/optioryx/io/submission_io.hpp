#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "optioryx/core/model.hpp"

namespace optioryx {

struct SubmissionRow {
    std::string wave_id;
    std::string order_id;
    std::string list_id;
};

std::vector<SubmissionRow> build_submission_rows(const ProblemInstance& instance, const Solution& solution);
void write_submission_csv(const std::filesystem::path& path, const ProblemInstance& instance, const Solution& solution);
std::vector<SubmissionRow> load_submission_csv(const std::filesystem::path& path);
std::vector<std::string> validate_submission_rows(const ProblemInstance& instance, const std::vector<SubmissionRow>& rows);
double score_submission_rows(const ProblemInstance& instance, const std::vector<SubmissionRow>& rows);

}  // namespace optioryx
