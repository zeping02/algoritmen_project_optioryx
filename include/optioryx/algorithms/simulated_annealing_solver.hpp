#pragma once

#include <random>
#include <string>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

struct SimulatedAnnealingConfig {
    std::string config_id = "sa_default";
    StartSolutionStrategy start_strategy = StartSolutionStrategy::greedy;
    // De greedy warm-start is zelf configureerbaar. Expliciete waarden voorkomen
    // dat een SA-vergelijking ongemerkt een sterkere greedy gebruikt dan haar baseline.
    int greedy_start_local_search_passes = 1;
    int greedy_start_local_search_neighbor_window = 8;
    NeighborhoodStrategy neighborhood = NeighborhoodStrategy::relocate;
    CoolingSchedule cooling = CoolingSchedule::geometric;
    int iterations = 400;
    double initial_temperature = 60.0;
    double final_temperature = 0.05;
    double geometric_alpha = 0.97;
    double lundy_beta = 0.01;
};

WaveSolveResult solve_wave_simulated_annealing(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const SimulatedAnnealingConfig& config,
    std::mt19937_64& rng);

}  // namespace optioryx
