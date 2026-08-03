#pragma once

#include <string>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

struct SeedSelectionConfig {
    std::string config_id = "seed_selection_default";
    SeedRule seed_rule = SeedRule::highest_standalone_cost;
    SelectionRule selection_rule = SelectionRule::minimum_incremental_cost;
};

WaveSolveResult solve_wave_seed_selection(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const SeedSelectionConfig& config);

}  // namespace optioryx
