#include "optioryx/core/evaluator.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace optioryx {

namespace {

std::vector<int> sort_and_deduplicate(const std::vector<int>& values) {
    std::vector<int> result = values;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}  // namespace

BatchEvaluator::BatchEvaluator(const ProblemInstance& instance)
    : instance_(instance), cache_(instance.waves.size()) {}

std::string BatchEvaluator::make_key(const std::vector<int>& order_indices) const {
    std::vector<int> canonical = sort_and_deduplicate(order_indices);
    std::ostringstream stream;
    for (std::size_t index = 0; index < canonical.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << canonical[index];
    }
    return stream.str();
}

BatchStats BatchEvaluator::compute_stats(int wave_index, const std::vector<int>& order_indices) const {
    if (wave_index < 0 || static_cast<std::size_t>(wave_index) >= instance_.waves.size()) {
        throw std::out_of_range("Wave index out of range");
    }

    std::vector<int> canonical_orders = sort_and_deduplicate(order_indices);
    std::vector<int> items;
    items.reserve(canonical_orders.size() * 3U);
    for (const int order_index : canonical_orders) {
        const Order& order = instance_.orders.at(static_cast<std::size_t>(order_index));
        items.insert(items.end(), order.items.begin(), order.items.end());
    }

    items = sort_and_deduplicate(items);
    items.erase(std::remove(items.begin(), items.end(), instance_.depot_index), items.end());

    BatchStats stats;
    stats.unique_item_count = items.size();
    if (items.empty()) {
        return stats;
    }

    std::sort(items.begin(), items.end(), [&](const int left, const int right) {
        return instance_.snake_rank.at(static_cast<std::size_t>(left)) <
               instance_.snake_rank.at(static_cast<std::size_t>(right));
    });

    stats.min_snake_rank = instance_.snake_rank.at(static_cast<std::size_t>(items.front()));
    stats.max_snake_rank = instance_.snake_rank.at(static_cast<std::size_t>(items.back()));
    stats.span_width = stats.max_snake_rank - stats.min_snake_rank;

    int previous = instance_.depot_index;
    for (const int item : items) {
        stats.cost += instance_.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
    }
    stats.cost += instance_.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance_.depot_index));
    return stats;
}

BatchStats BatchEvaluator::batch_stats(int wave_index, const std::vector<int>& order_indices) const {
    const std::string key = make_key(order_indices);
    auto& wave_cache = cache_.at(static_cast<std::size_t>(wave_index));
    const auto found = wave_cache.find(key);
    if (found != wave_cache.end()) {
        return found->second;
    }

    BatchStats stats = compute_stats(wave_index, order_indices);
    wave_cache.emplace(key, stats);
    return stats;
}

double BatchEvaluator::batch_cost(int wave_index, const std::vector<int>& order_indices) const {
    return batch_stats(wave_index, order_indices).cost;
}

double BatchEvaluator::wave_cost(int wave_index, const std::vector<Batch>& batches) const {
    double total_cost = 0.0;
    for (const Batch& batch : batches) {
        total_cost += batch_cost(wave_index, batch.order_indices);
    }
    return total_cost;
}

double BatchEvaluator::solution_cost(const Solution& solution) const {
    double total_cost = 0.0;
    for (const WaveSolution& wave_solution : solution.wave_solutions) {
        const auto wave_it = instance_.wave_index_by_id.find(wave_solution.wave_id);
        if (wave_it == instance_.wave_index_by_id.end()) {
            continue;
        }
        total_cost += wave_cost(wave_it->second, wave_solution.batches);
    }
    return total_cost;
}

int BatchEvaluator::order_span_width(int order_index) const {
    const Order& order = instance_.orders.at(static_cast<std::size_t>(order_index));
    if (order.items.empty()) {
        return 0;
    }

    int min_rank = std::numeric_limits<int>::max();
    int max_rank = std::numeric_limits<int>::min();
    for (const int item : order.items) {
        min_rank = std::min(min_rank, instance_.snake_rank.at(static_cast<std::size_t>(item)));
        max_rank = std::max(max_rank, instance_.snake_rank.at(static_cast<std::size_t>(item)));
    }
    return max_rank - min_rank;
}

std::vector<std::string> BatchEvaluator::validate_solution(const Solution& solution) const {
    std::vector<std::string> errors;
    std::vector<int> assignments(instance_.orders.size(), 0);

    for (const WaveSolution& wave_solution : solution.wave_solutions) {
        const auto wave_it = instance_.wave_index_by_id.find(wave_solution.wave_id);
        if (wave_it == instance_.wave_index_by_id.end()) {
            errors.push_back("Unknown wave in solution: " + wave_solution.wave_id);
            continue;
        }

        const Wave& wave = instance_.waves.at(static_cast<std::size_t>(wave_it->second));
        for (const Batch& batch : wave_solution.batches) {
            if (batch.wave_id != wave_solution.wave_id) {
                errors.push_back("Batch wave id mismatch in wave " + wave_solution.wave_id);
            }
            if (batch.order_indices.size() > instance_.nmax) {
                errors.push_back("Batch in wave " + wave_solution.wave_id + " exceeds nmax");
            }
            for (const int order_index : batch.order_indices) {
                if (order_index < 0 || static_cast<std::size_t>(order_index) >= instance_.orders.size()) {
                    errors.push_back("Invalid order index in wave " + wave_solution.wave_id);
                    continue;
                }
                const Order& order = instance_.orders.at(static_cast<std::size_t>(order_index));
                if (order.wave_id != wave_solution.wave_id) {
                    errors.push_back("Order " + order.id + " assigned to wrong wave");
                }
                if (std::find(wave.order_indices.begin(), wave.order_indices.end(), order_index) == wave.order_indices.end()) {
                    errors.push_back("Order " + order.id + " is not part of wave " + wave_solution.wave_id);
                }
                assignments.at(static_cast<std::size_t>(order_index)) += 1;
            }
        }
    }

    for (std::size_t order_index = 0; order_index < assignments.size(); ++order_index) {
        if (assignments[order_index] != 1) {
            errors.push_back(
                "Order " + instance_.orders[order_index].id + " assigned " + std::to_string(assignments[order_index]) + " times");
        }
    }

    return errors;
}

}  // namespace optioryx
