#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace optioryx {

struct Order {
    std::string id;
    std::string wave_id;
    std::vector<int> items;
};

struct Wave {
    std::string id;
    std::vector<int> order_indices;
};

struct Batch {
    std::string wave_id;
    std::vector<int> order_indices;
};

struct ConvergencePoint {
    int step = 0;
    double best_objective = 0.0;
    double current_objective = 0.0;
    double control_value = 0.0;
};

struct WaveSolution {
    std::string wave_id;
    std::vector<Batch> batches;
    double objective = 0.0;
    double runtime_ms = 0.0;
    std::vector<ConvergencePoint> convergence;
    std::map<std::string, std::string> metadata;
};

struct Solution {
    std::string solver_name;
    std::string config_id;
    std::uint64_t seed = 0;
    std::vector<WaveSolution> wave_solutions;
    double objective = 0.0;
    double runtime_ms = 0.0;
    bool feasible = false;
    std::vector<std::string> validation_errors;
};

struct ProblemInstance {
    std::string name;
    std::size_t nmax = 0;
    int start_index = 0;
    int end_index = 0;
    std::vector<std::string> item_ids;
    std::vector<int> snake_rank;
    std::vector<std::vector<double>> distances;
    std::vector<Order> orders;
    std::vector<Wave> waves;
    std::unordered_map<std::string, int> item_index_by_id;
    std::unordered_map<std::string, int> order_index_by_id;
    std::unordered_map<std::string, int> wave_index_by_id;
};

struct WaveMetrics {
    double overlap_density = 0.0;
    double span_dispersion = 0.0;
};

}  // namespace optioryx
