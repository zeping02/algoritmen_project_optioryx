#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "optioryx/core/model.hpp"

namespace optioryx {

// Startoplossing voor metaheuristieken.
// greedy gebruikt de beste deterministische constructie als warm-start;
// randomized maakt expres variatie voor experimenten.
enum class StartSolutionStrategy {
    greedy,
    randomized,
};

// Moves die simulated annealing kan gebruiken om een buuroplossing te maken.
enum class NeighborhoodStrategy {
    relocate,
    swap,
    merge_split,
    mixed,
};

// Temperatuurschema's voor simulated annealing.
enum class CoolingSchedule {
    geometric,
    lundy_mees,
};

// Regel om een nieuwe seed-order te kiezen in de seed-selection baseline.
enum class SeedRule {
    highest_standalone_cost,
    largest_snake_span,
};

// Regel om de volgende order aan een seed-batch toe te voegen.
enum class SelectionRule {
    minimum_incremental_cost,
    minimum_span_expansion,
    hybrid_score,
};

// Uniform resultaat dat alle wave-solvers teruggeven.
struct WaveSolveResult {
    std::vector<Batch> batches;
    double objective = 0.0;
    double runtime_ms = 0.0;
    std::vector<ConvergencePoint> convergence;
    std::map<std::string, std::string> metadata;
};

// Een record per benchmarkrun en wave. Deze records worden naar CSV geschreven
// zodat scripts en de paper dezelfde meetdata kunnen gebruiken.
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

// Convergentiedata voor iteratieve algoritmes.
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

// Parse- en formatteerfuncties houden configbestanden leesbaar zonder strings
// door de algoritmecode zelf te verspreiden.
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
