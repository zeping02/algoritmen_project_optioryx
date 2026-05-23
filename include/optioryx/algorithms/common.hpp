#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "optioryx/core/model.hpp"

namespace optioryx {

enum class StartSolutionStrategy {
    greedy,
    randomized,
};

enum class NeighborhoodStrategy {
    relocate,
    swap,
    merge_split,
    mixed,
};

enum class CoolingSchedule {
    geometric,
    lundy_mees,
};

enum class SeedRule {
    highest_standalone_cost,
    largest_snake_span,
};

enum class SelectionRule {
    minimum_incremental_cost,
    minimum_span_expansion,
    hybrid_score,
};

struct WaveSolveResult {
    std::vector<Batch> batches;
    double objective = 0.0;
    double runtime_ms = 0.0;
    std::vector<ConvergencePoint> convergence;
    std::map<std::string, std::string> metadata;
};

struct BenchmarkRecord {
    std::string solver;
    std::string config_id;
    std::string wave_id;
    std::uint64_t seed = 0;
    int order_count = 0;
    double objective = 0.0;
    double runtime_ms = 0.0;
    double overlap_density = 0.0;
    double span_dispersion = 0.0;
};

struct ConvergenceRecord {
    std::string solver;
    std::string config_id;
    std::string wave_id;
    std::uint64_t seed = 0;
    int step = 0;
    double best_objective = 0.0;
    double current_objective = 0.0;
    double control_value = 0.0;
};

StartSolutionStrategy parse_start_strategy(const std::string& value);
NeighborhoodStrategy parse_neighborhood_strategy(const std::string& value);
CoolingSchedule parse_cooling_schedule(const std::string& value);
SeedRule parse_seed_rule(const std::string& value);
SelectionRule parse_selection_rule(const std::string& value);

std::string to_string(StartSolutionStrategy value);
std::string to_string(NeighborhoodStrategy value);
std::string to_string(CoolingSchedule value);
std::string to_string(SeedRule value);
std::string to_string(SelectionRule value);

}  // namespace optioryx
