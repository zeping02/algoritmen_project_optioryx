#pragma once

#include <string>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

struct GreedyConfig {
    std::string config_id = "greedy_default";
    int local_search_passes = 1;
    int local_search_neighbor_window = 8;
    int route_elimination_passes = 0;
    int route_elimination_max_batch_size = 12;
    int route_elimination_beam_width = 1;
    int route_elimination_candidates_per_order = 1;
    int route_elimination_ordering_trials = 1;
    int post_elimination_local_search_passes = 0;
    int pair_rebuild_passes = 0;
    int pair_rebuild_neighbor_window = 0;
    int triple_rebuild_passes = 0;
    int triple_rebuild_neighbor_window = 0;
    int triple_rebuild_combination_window = 0;
    int post_triple_pair_rebuild_passes = 0;
    int post_triple_pair_rebuild_neighbor_window = 0;
    int post_pair_triple_rebuild_passes = 0;
    int post_pair_triple_rebuild_neighbor_window = 0;
    int post_pair_triple_rebuild_combination_window = 0;
    int final_local_search_passes = 0;
};

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config);

}  // namespace optioryx
