#include "optioryx/algorithms/greedy_solver.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace optioryx {

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config) {
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    std::vector<Batch> batches;
    batches.reserve(wave.order_indices.size());
    for (const int order_index : wave.order_indices) {
        Batch batch;
        batch.wave_id = wave.id;
        batch.order_indices.push_back(order_index);
        batches.push_back(batch);
    }

    while (true) {
        double best_saving = 0.0;
        double best_merged_cost = std::numeric_limits<double>::infinity();
        std::size_t best_left = 0;
        std::size_t best_right = 0;
        bool found = false;

        for (std::size_t left = 0; left < batches.size(); ++left) {
            for (std::size_t right = left + 1; right < batches.size(); ++right) {
                if (batches[left].order_indices.size() + batches[right].order_indices.size() > instance.nmax) {
                    continue;
                }

                std::vector<int> merged_orders = batches[left].order_indices;
                merged_orders.insert(
                    merged_orders.end(),
                    batches[right].order_indices.begin(),
                    batches[right].order_indices.end());

                const double merged_cost = evaluator.batch_cost(wave_index, merged_orders);
                const double saving = evaluator.batch_cost(wave_index, batches[left].order_indices) +
                                      evaluator.batch_cost(wave_index, batches[right].order_indices) - merged_cost;

                if (saving > best_saving + 1e-9 ||
                    (!found && saving > 0.0) ||
                    (std::abs(saving - best_saving) <= 1e-9 && merged_cost < best_merged_cost)) {
                    best_saving = saving;
                    best_merged_cost = merged_cost;
                    best_left = left;
                    best_right = right;
                    found = saving > 0.0;
                }
            }
        }

        if (!found) {
            break;
        }

        batches[best_left].order_indices.insert(
            batches[best_left].order_indices.end(),
            batches[best_right].order_indices.begin(),
            batches[best_right].order_indices.end());
        batches.erase(batches.begin() + static_cast<std::ptrdiff_t>(best_right));
    }

    WaveSolveResult result;
    result.batches = std::move(batches);
    result.objective = evaluator.wave_cost(wave_index, result.batches);
    result.metadata["construction"] = "savings_merge";
    result.metadata["greedy_choice"] = "largest_positive_pairwise_saving";
    result.metadata["config_id"] = config.config_id;

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
