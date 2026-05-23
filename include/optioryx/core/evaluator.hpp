#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "optioryx/core/model.hpp"

namespace optioryx {

struct BatchStats {
    double cost = 0.0;
    int min_snake_rank = 0;
    int max_snake_rank = 0;
    int span_width = 0;
    std::size_t unique_item_count = 0;
};

class BatchEvaluator {
public:
    explicit BatchEvaluator(const ProblemInstance& instance);

    BatchStats batch_stats(int wave_index, const std::vector<int>& order_indices) const;
    double batch_cost(int wave_index, const std::vector<int>& order_indices) const;
    double wave_cost(int wave_index, const std::vector<Batch>& batches) const;
    double solution_cost(const Solution& solution) const;
    std::vector<std::string> validate_solution(const Solution& solution) const;

    int order_span_width(int order_index) const;

private:
    std::string make_key(const std::vector<int>& order_indices) const;
    BatchStats compute_stats(int wave_index, const std::vector<int>& order_indices) const;

    const ProblemInstance& instance_;
    std::size_t mask_words_ = 0;
    std::vector<int> items_by_snake_rank_;
    std::vector<std::vector<std::uint64_t>> order_item_masks_;
    std::vector<std::vector<int>> order_items_by_snake_rank_;
    mutable std::vector<std::unordered_map<std::string, BatchStats>> cache_;
};

}  // namespace optioryx
