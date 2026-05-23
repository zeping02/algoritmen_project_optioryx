#pragma once

#include <string>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

struct GreedyConfig {
    std::string config_id = "greedy_default";
    int local_search_passes = 1;
    int local_search_neighbor_window = 8;
};

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config);

}  // namespace optioryx
