#pragma once

#include <random>
#include <string>
#include <vector>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

struct GeneticConfig {
    std::string config_id = "ga_default";
    int population_size = 40;
    int generations = 120;
    int tournament_size = 3;
    double mutation_rate = 0.2;
    double insertion_rate = 0.5;
    double elitism_fraction = 0.05;
};

struct DecodeResult {
    std::vector<Batch> batches;
    double objective = 0.0;
};

DecodeResult decode_permutation_to_batches(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& permutation);

WaveSolveResult solve_wave_genetic(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GeneticConfig& config,
    std::mt19937_64& rng);

}  // namespace optioryx
