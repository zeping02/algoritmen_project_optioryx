#include "optioryx/algorithms/greedy_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace optioryx {

namespace {

struct ActiveBatch {
    Batch batch;
    std::vector<std::uint64_t> item_mask;
    double cost = 0.0;
    int version = 0;
    bool active = true;
};

struct SearchBatch {
    Batch batch;
    std::vector<std::uint16_t> item_counts;
    double cost = 0.0;
    int min_rank = 0;
    int max_rank = 0;
};

struct MergeCandidate {
    double saving = 0.0;
    double merged_cost = 0.0;
    std::size_t left_id = 0;
    std::size_t right_id = 0;
    int left_version = 0;
    int right_version = 0;
};

enum class LocalActionType {
    none,
    swap_orders,
    move_left_to_right,
    move_right_to_left,
};

struct LocalAction {
    LocalActionType type = LocalActionType::none;
    std::size_t left_pos = 0;
    std::size_t right_pos = 0;
    double delta = 0.0;
};

struct MergeCandidateLess {
    bool operator()(const MergeCandidate& left, const MergeCandidate& right) const {
        if (std::abs(left.saving - right.saving) > 1e-9) {
            return left.saving < right.saving;
        }
        if (std::abs(left.merged_cost - right.merged_cost) > 1e-9) {
            return left.merged_cost > right.merged_cost;
        }
        return std::tie(left.left_id, left.right_id) > std::tie(right.left_id, right.right_id);
    }
};

std::vector<int> build_items_by_snake_rank(const ProblemInstance& instance) {
    std::vector<int> items(instance.item_ids.size());
    std::iota(items.begin(), items.end(), 0);
    std::sort(items.begin(), items.end(), [&](const int left, const int right) {
        return instance.snake_rank.at(static_cast<std::size_t>(left)) <
               instance.snake_rank.at(static_cast<std::size_t>(right));
    });
    return items;
}

std::vector<std::uint64_t> build_order_mask(
    const ProblemInstance& instance,
    int order_index,
    std::size_t mask_words) {
    std::vector<std::uint64_t> mask(mask_words, 0U);
    const Order& order = instance.orders.at(static_cast<std::size_t>(order_index));
    for (const int item : order.items) {
        if (item == instance.start_index || item == instance.end_index) {
            continue;
        }
        const std::size_t item_index = static_cast<std::size_t>(item);
        mask[item_index / 64U] |= (std::uint64_t{1} << (item_index % 64U));
    }
    return mask;
}

std::vector<std::vector<int>> build_order_items_by_snake_rank(const ProblemInstance& instance) {
    std::vector<std::vector<int>> order_items(instance.orders.size());
    for (std::size_t order_index = 0; order_index < instance.orders.size(); ++order_index) {
        std::vector<int> items = instance.orders[order_index].items;
        items.erase(std::remove(items.begin(), items.end(), instance.start_index), items.end());
        items.erase(std::remove(items.begin(), items.end(), instance.end_index), items.end());
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
        std::sort(items.begin(), items.end(), [&](const int left, const int right) {
            return instance.snake_rank.at(static_cast<std::size_t>(left)) <
                   instance.snake_rank.at(static_cast<std::size_t>(right));
        });
        order_items[order_index] = std::move(items);
    }
    return order_items;
}

bool mask_contains(const std::vector<std::uint64_t>& mask, int item) {
    const std::size_t item_index = static_cast<std::size_t>(item);
    return (mask[item_index / 64U] & (std::uint64_t{1} << (item_index % 64U))) != 0U;
}

double route_cost_for_mask(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint64_t>& mask) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    for (const int item : items_by_snake_rank) {
        if (!mask_contains(mask, item)) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

double route_cost_for_counts(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint16_t>& item_counts) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    for (const int item : items_by_snake_rank) {
        if (item_counts[static_cast<std::size_t>(item)] == 0U) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

void update_search_batch_stats(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    SearchBatch& batch) {
    batch.cost = 0.0;
    batch.min_rank = 0;
    batch.max_rank = 0;

    int previous = instance.start_index;
    bool has_items = false;
    for (const int item : items_by_snake_rank) {
        if (batch.item_counts[static_cast<std::size_t>(item)] == 0U) {
            continue;
        }
        if (!has_items) {
            batch.min_rank = instance.snake_rank.at(static_cast<std::size_t>(item));
            has_items = true;
        }
        batch.max_rank = instance.snake_rank.at(static_cast<std::size_t>(item));
        batch.cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
    }
    if (has_items) {
        batch.cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    }
}

double merged_route_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint64_t>& left_mask,
    const std::vector<std::uint64_t>& right_mask) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    for (const int item : items_by_snake_rank) {
        const std::size_t item_index = static_cast<std::size_t>(item);
        const std::uint64_t bit = std::uint64_t{1} << (item_index % 64U);
        const bool present = ((left_mask[item_index / 64U] | right_mask[item_index / 64U]) & bit) != 0U;
        if (!present) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

void merge_mask_into(std::vector<std::uint64_t>& target, const std::vector<std::uint64_t>& source) {
    for (std::size_t word = 0; word < target.size(); ++word) {
        target[word] |= source[word];
    }
}

void apply_order_count_delta(
    std::vector<std::uint16_t>& item_counts,
    const std::vector<int>& order_items,
    int delta) {
    for (const int item : order_items) {
        std::uint16_t& count = item_counts[static_cast<std::size_t>(item)];
        if (delta > 0) {
            ++count;
        } else {
            --count;
        }
    }
}

std::vector<std::uint16_t> build_item_counts(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<int>& order_indices) {
    std::vector<std::uint16_t> counts(instance.item_ids.size(), 0U);
    for (const int order_index : order_indices) {
        apply_order_count_delta(counts, order_items_by_snake_rank[static_cast<std::size_t>(order_index)], 1);
    }
    return counts;
}

std::vector<SearchBatch> build_search_batches(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<Batch>& batches) {
    std::vector<SearchBatch> search_batches;
    search_batches.reserve(batches.size());
    for (const Batch& batch : batches) {
        SearchBatch search_batch;
        search_batch.batch = batch;
        search_batch.item_counts = build_item_counts(instance, order_items_by_snake_rank, batch.order_indices);
        update_search_batch_stats(instance, items_by_snake_rank, search_batch);
        search_batches.push_back(std::move(search_batch));
    }
    return search_batches;
}

double span_center(const SearchBatch& batch) {
    return (static_cast<double>(batch.min_rank) + static_cast<double>(batch.max_rank)) / 2.0;
}

double simulated_swap_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right,
    std::size_t left_pos,
    std::size_t right_pos) {
    const int left_order = left.batch.order_indices[left_pos];
    const int right_order = right.batch.order_indices[right_pos];
    const std::vector<int>& left_items = order_items_by_snake_rank[static_cast<std::size_t>(left_order)];
    const std::vector<int>& right_items = order_items_by_snake_rank[static_cast<std::size_t>(right_order)];

    apply_order_count_delta(left.item_counts, left_items, -1);
    apply_order_count_delta(left.item_counts, right_items, 1);
    apply_order_count_delta(right.item_counts, right_items, -1);
    apply_order_count_delta(right.item_counts, left_items, 1);
    const double cost = route_cost_for_counts(instance, items_by_snake_rank, left.item_counts) +
                        route_cost_for_counts(instance, items_by_snake_rank, right.item_counts);
    apply_order_count_delta(left.item_counts, right_items, -1);
    apply_order_count_delta(left.item_counts, left_items, 1);
    apply_order_count_delta(right.item_counts, left_items, -1);
    apply_order_count_delta(right.item_counts, right_items, 1);
    return cost;
}

double simulated_move_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& source,
    SearchBatch& destination,
    std::size_t source_pos) {
    const int moved_order = source.batch.order_indices[source_pos];
    const std::vector<int>& moved_items = order_items_by_snake_rank[static_cast<std::size_t>(moved_order)];

    apply_order_count_delta(source.item_counts, moved_items, -1);
    apply_order_count_delta(destination.item_counts, moved_items, 1);
    const double cost = route_cost_for_counts(instance, items_by_snake_rank, source.item_counts) +
                        route_cost_for_counts(instance, items_by_snake_rank, destination.item_counts);
    apply_order_count_delta(source.item_counts, moved_items, 1);
    apply_order_count_delta(destination.item_counts, moved_items, -1);
    return cost;
}

LocalAction find_best_pair_action(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right) {
    LocalAction best;
    const double current_cost = left.cost + right.cost;

    for (std::size_t left_pos = 0; left_pos < left.batch.order_indices.size(); ++left_pos) {
        for (std::size_t right_pos = 0; right_pos < right.batch.order_indices.size(); ++right_pos) {
            const double candidate_cost = simulated_swap_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, left, right, left_pos, right_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::swap_orders, left_pos, right_pos, delta};
            }
        }
    }

    if (right.batch.order_indices.size() < instance.nmax && left.batch.order_indices.size() > 1U) {
        for (std::size_t left_pos = 0; left_pos < left.batch.order_indices.size(); ++left_pos) {
            const double candidate_cost = simulated_move_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, left, right, left_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::move_left_to_right, left_pos, 0, delta};
            }
        }
    }

    if (left.batch.order_indices.size() < instance.nmax && right.batch.order_indices.size() > 1U) {
        for (std::size_t right_pos = 0; right_pos < right.batch.order_indices.size(); ++right_pos) {
            const double candidate_cost = simulated_move_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, right, left, right_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::move_right_to_left, 0, right_pos, delta};
            }
        }
    }

    return best;
}

void apply_local_action(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right,
    const LocalAction& action) {
    if (action.type == LocalActionType::swap_orders) {
        const int left_order = left.batch.order_indices[action.left_pos];
        const int right_order = right.batch.order_indices[action.right_pos];
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(left_order)], -1);
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(right_order)], 1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(right_order)], -1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(left_order)], 1);
        std::swap(left.batch.order_indices[action.left_pos], right.batch.order_indices[action.right_pos]);
    } else if (action.type == LocalActionType::move_left_to_right) {
        const int moved_order = left.batch.order_indices[action.left_pos];
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], -1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], 1);
        left.batch.order_indices.erase(left.batch.order_indices.begin() + static_cast<std::ptrdiff_t>(action.left_pos));
        right.batch.order_indices.push_back(moved_order);
    } else if (action.type == LocalActionType::move_right_to_left) {
        const int moved_order = right.batch.order_indices[action.right_pos];
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], -1);
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], 1);
        right.batch.order_indices.erase(right.batch.order_indices.begin() + static_cast<std::ptrdiff_t>(action.right_pos));
        left.batch.order_indices.push_back(moved_order);
    }

    update_search_batch_stats(instance, items_by_snake_rank, left);
    update_search_batch_stats(instance, items_by_snake_rank, right);
}

std::size_t improve_with_local_search(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<Batch>& batches,
    int passes,
    int neighbor_window) {
    if (passes <= 0 || neighbor_window <= 0 || batches.size() < 2U) {
        return 0;
    }

    std::vector<SearchBatch> search_batches =
        build_search_batches(instance, items_by_snake_rank, order_items_by_snake_rank, batches);
    std::size_t improvement_count = 0;

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<std::size_t> sorted_indices(search_batches.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const std::size_t left, const std::size_t right) {
            return span_center(search_batches[left]) < span_center(search_batches[right]);
        });

        bool improved_this_pass = false;
        for (std::size_t pos = 0; pos < sorted_indices.size(); ++pos) {
            const std::size_t max_neighbor =
                std::min(sorted_indices.size(), pos + 1U + static_cast<std::size_t>(neighbor_window));
            for (std::size_t neighbor_pos = pos + 1U; neighbor_pos < max_neighbor; ++neighbor_pos) {
                SearchBatch& left = search_batches[sorted_indices[pos]];
                SearchBatch& right = search_batches[sorted_indices[neighbor_pos]];
                while (true) {
                    const LocalAction action =
                        find_best_pair_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right);
                    if (action.type == LocalActionType::none) {
                        break;
                    }
                    apply_local_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right, action);
                    ++improvement_count;
                    improved_this_pass = true;
                }
            }
        }

        if (!improved_this_pass) {
            break;
        }
    }

    batches.clear();
    batches.reserve(search_batches.size());
    for (SearchBatch& batch : search_batches) {
        batches.push_back(std::move(batch.batch));
    }
    return improvement_count;
}

void push_merge_candidate(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<ActiveBatch>& batches,
    std::size_t left_id,
    std::size_t right_id,
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateLess>& candidates) {
    const ActiveBatch& left = batches[left_id];
    const ActiveBatch& right = batches[right_id];
    if (!left.active || !right.active) {
        return;
    }
    if (left.batch.order_indices.size() + right.batch.order_indices.size() > instance.nmax) {
        return;
    }

    const double merged_cost = merged_route_cost(instance, items_by_snake_rank, left.item_mask, right.item_mask);
    const double saving = left.cost + right.cost - merged_cost;
    if (saving <= 1e-9) {
        return;
    }

    candidates.push({saving, merged_cost, left_id, right_id, left.version, right.version});
}

bool is_current_candidate(const std::vector<ActiveBatch>& batches, const MergeCandidate& candidate) {
    if (candidate.left_id >= batches.size() || candidate.right_id >= batches.size()) {
        return false;
    }
    const ActiveBatch& left = batches[candidate.left_id];
    const ActiveBatch& right = batches[candidate.right_id];
    return left.active && right.active &&
           left.version == candidate.left_version &&
           right.version == candidate.right_version;
}

}  // namespace

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config) {
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));
    const std::size_t mask_words = (instance.item_ids.size() + 63U) / 64U;
    const std::vector<int> items_by_snake_rank = build_items_by_snake_rank(instance);

    std::vector<ActiveBatch> batches;
    batches.reserve(wave.order_indices.size());
    for (const int order_index : wave.order_indices) {
        Batch batch;
        batch.wave_id = wave.id;
        batch.order_indices.push_back(order_index);
        ActiveBatch active_batch;
        active_batch.item_mask = build_order_mask(instance, order_index, mask_words);
        active_batch.cost = route_cost_for_mask(instance, items_by_snake_rank, active_batch.item_mask);
        active_batch.batch = std::move(batch);
        batches.push_back(std::move(active_batch));
    }

    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateLess> candidates;
    for (std::size_t left = 0; left < batches.size(); ++left) {
        for (std::size_t right = left + 1U; right < batches.size(); ++right) {
            push_merge_candidate(instance, items_by_snake_rank, batches, left, right, candidates);
        }
    }

    std::size_t active_count = batches.size();
    std::size_t merge_count = 0;
    while (!candidates.empty()) {
        const MergeCandidate candidate = candidates.top();
        candidates.pop();
        if (!is_current_candidate(batches, candidate)) {
            continue;
        }

        ActiveBatch& left = batches[candidate.left_id];
        ActiveBatch& right = batches[candidate.right_id];
        left.batch.order_indices.insert(
            left.batch.order_indices.end(),
            right.batch.order_indices.begin(),
            right.batch.order_indices.end());
        merge_mask_into(left.item_mask, right.item_mask);
        left.cost = candidate.merged_cost;
        ++left.version;
        right.active = false;
        ++right.version;
        --active_count;
        ++merge_count;

        for (std::size_t other_id = 0; other_id < batches.size(); ++other_id) {
            if (other_id == candidate.left_id || !batches[other_id].active) {
                continue;
            }
            const std::size_t next_left = std::min(candidate.left_id, other_id);
            const std::size_t next_right = std::max(candidate.left_id, other_id);
            push_merge_candidate(instance, items_by_snake_rank, batches, next_left, next_right, candidates);
        }
    }

    std::vector<Batch> result_batches;
    result_batches.reserve(active_count);
    for (const ActiveBatch& batch : batches) {
        if (batch.active) {
            result_batches.push_back(batch.batch);
        }
    }

    std::size_t local_improvements = 0;
    if (config.local_search_passes > 0 && config.local_search_neighbor_window > 0) {
        const std::vector<std::vector<int>> order_items_by_snake_rank = build_order_items_by_snake_rank(instance);
        local_improvements = improve_with_local_search(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            result_batches,
            config.local_search_passes,
            config.local_search_neighbor_window);
    }

    WaveSolveResult result;
    result.batches = std::move(result_batches);
    result.objective = evaluator.wave_cost(wave_index, result.batches);
    result.metadata["construction"] = "savings_merge";
    result.metadata["greedy_choice"] = "largest_positive_pairwise_saving";
    result.metadata["config_id"] = config.config_id;
    result.metadata["merge_updates"] = "lazy_priority_queue";
    result.metadata["merge_count"] = std::to_string(merge_count);
    result.metadata["local_search_passes"] = std::to_string(config.local_search_passes);
    result.metadata["local_search_neighbor_window"] = std::to_string(config.local_search_neighbor_window);
    result.metadata["local_improvements"] = std::to_string(local_improvements);

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
