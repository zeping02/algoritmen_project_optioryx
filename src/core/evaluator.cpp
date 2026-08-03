#include "optioryx/core/evaluator.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace optioryx {

namespace {

// Veel berekeningen hangen alleen af van de verzameling orders in een batch,
// niet van hun volgorde. Daarom sorteren en dedupliceren we op centrale plaatsen.
std::vector<int> sort_and_deduplicate(const std::vector<int>& values) {
    std::vector<int> result = values;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}  // namespace

BatchEvaluator::BatchEvaluator(const ProblemInstance& instance, const std::size_t cache_entry_limit)
    : instance_(instance),
      mask_words_((instance.item_ids.size() + 63U) / 64U),
      items_by_snake_rank_(instance.item_ids.size()),
      order_item_masks_(instance.orders.size(), std::vector<std::uint64_t>(mask_words_, 0U)),
      order_items_by_snake_rank_(instance.orders.size()),
      item_mask_scratch_(mask_words_, 0U),
      cache_entry_limit_(cache_entry_limit),
      cache_(instance.waves.size()) {
    // Voor scoring moet elke batch in de vaste snake-volgorde worden doorlopen.
    // Deze lijst voorkomt dat we die volgorde bij elke routekost opnieuw moeten sorteren.
    std::iota(items_by_snake_rank_.begin(), items_by_snake_rank_.end(), 0);
    std::sort(items_by_snake_rank_.begin(), items_by_snake_rank_.end(), [&](const int left, const int right) {
        return instance_.snake_rank.at(static_cast<std::size_t>(left)) <
               instance_.snake_rank.at(static_cast<std::size_t>(right));
    });
    std::vector<std::size_t> snake_position_by_item(instance_.item_ids.size(), 0U);
    for (std::size_t snake_position = 0; snake_position < items_by_snake_rank_.size(); ++snake_position) {
        snake_position_by_item[static_cast<std::size_t>(items_by_snake_rank_[snake_position])] = snake_position;
    }

    for (std::size_t order_index = 0; order_index < instance_.orders.size(); ++order_index) {
        // Per order bewaren we twee vormen:
        // - een gesorteerde lijst van unieke items voor span-berekeningen;
        // - een bitmask voor snelle union van items wanneer meerdere orders een batch vormen.
        std::vector<int> unique_items = sort_and_deduplicate(instance_.orders[order_index].items);
        std::sort(unique_items.begin(), unique_items.end(), [&](const int left, const int right) {
            return instance_.snake_rank.at(static_cast<std::size_t>(left)) <
                   instance_.snake_rank.at(static_cast<std::size_t>(right));
        });
        order_items_by_snake_rank_[order_index] = unique_items;

        for (const int item : unique_items) {
            if (item == instance_.start_index || item == instance_.end_index) {
                continue;
            }
            const std::size_t snake_position = snake_position_by_item[static_cast<std::size_t>(item)];
            order_item_masks_[order_index][snake_position / 64U] |=
                (std::uint64_t{1} << (snake_position % 64U));
        }
    }
}

std::string_view BatchEvaluator::make_key_view(const std::vector<int>& sorted_unique_order_indices) const {
    // De key is de compacte binaire representatie van de gesorteerde indices.
    // unordered_map vergelijkt de bytes tijdens find(); alleen een miss krijgt
    // verderop een owning std::string.
    if (sorted_unique_order_indices.empty()) {
        return {};
    }
    return {
        reinterpret_cast<const char*>(sorted_unique_order_indices.data()),
        sorted_unique_order_indices.size() * sizeof(int)};
}

BatchStats BatchEvaluator::compute_stats_sorted_unique(
    int wave_index,
    const std::vector<int>& sorted_unique_order_indices) const {
    if (wave_index < 0 || static_cast<std::size_t>(wave_index) >= instance_.waves.size()) {
        throw std::out_of_range("Wave index out of range");
    }

    std::fill(item_mask_scratch_.begin(), item_mask_scratch_.end(), 0U);
    // Union van alle picklocaties in de batch. Als twee orders dezelfde locatie
    // nodig hebben, wordt die locatie maar een keer gepickt en dus maar een keer gescoord.
    for (const int order_index : sorted_unique_order_indices) {
        const auto& order_mask = order_item_masks_.at(static_cast<std::size_t>(order_index));
        for (std::size_t word = 0; word < mask_words_; ++word) {
            item_mask_scratch_[word] |= order_mask[word];
        }
    }

    BatchStats stats;
    int previous = instance_.start_index;
    bool has_items = false;
    // Score exact zoals de opdracht verwacht: START -> unieke locaties in
    // snake-volgorde -> END. Tegelijk vullen we extra statistieken voor analyse.
    for (std::size_t word = 0; word < item_mask_scratch_.size(); ++word) {
        std::uint64_t present_items = item_mask_scratch_[word];
        while (present_items != 0U) {
            const unsigned bit = std::countr_zero(present_items);
            const std::size_t snake_position = word * 64U + static_cast<std::size_t>(bit);
            const int item = items_by_snake_rank_[snake_position];
            const std::size_t item_index = static_cast<std::size_t>(item);
            if (!has_items) {
                stats.min_snake_rank = instance_.snake_rank.at(item_index);
                has_items = true;
            }
            stats.max_snake_rank = instance_.snake_rank.at(item_index);
            ++stats.unique_item_count;
            stats.cost +=
                instance_.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
            previous = item;
            present_items &= present_items - 1U;
        }
    }
    if (!has_items) {
        return stats;
    }

    stats.span_width = stats.max_snake_rank - stats.min_snake_rank;
    stats.cost += instance_.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance_.end_index));
    return stats;
}

BatchStats BatchEvaluator::batch_stats(int wave_index, const std::vector<int>& order_indices) const {
    return batch_stats_sorted_unique(wave_index, sort_and_deduplicate(order_indices));
}

BatchStats BatchEvaluator::batch_stats_sorted_unique(
    int wave_index,
    const std::vector<int>& order_indices) const {
    if (cache_entry_limit_ == 0U) {
        return compute_stats_sorted_unique(wave_index, order_indices);
    }

    const std::string_view key = make_key_view(order_indices);
    auto& wave_cache = cache_.at(static_cast<std::size_t>(wave_index));
    const auto found = wave_cache.find(key);
    if (found != wave_cache.end()) {
        return found->second;
    }

    // Local search vraagt vaak herhaaldelijk dezelfde batchkost op.
    // Caching maakt vooral pair/triple rebuilds veel goedkoper.
    BatchStats stats = compute_stats_sorted_unique(wave_index, order_indices);
    if (cache_size_ >= cache_entry_limit_) {
        // clear() alleen behoudt de bucket-arrays van unordered_map en kan dus
        // na opeenvolgende waves nog veel geheugen vasthouden. Swappen met een
        // lege map geeft ook die capaciteit daadwerkelijk terug.
        for (auto& cache_to_clear : cache_) {
            WaveCache{}.swap(cache_to_clear);
        }
        cache_size_ = 0U;
    }

    auto& target_cache = cache_.at(static_cast<std::size_t>(wave_index));
    const auto [unused, inserted] = target_cache.emplace(std::string(key), stats);
    static_cast<void>(unused);
    if (inserted) {
        ++cache_size_;
    }
    return stats;
}

std::size_t BatchEvaluator::cache_size() const noexcept {
    return cache_size_;
}

double BatchEvaluator::batch_cost(int wave_index, const std::vector<int>& order_indices) const {
    return batch_stats(wave_index, order_indices).cost;
}

double BatchEvaluator::batch_cost_sorted_unique(int wave_index, const std::vector<int>& order_indices) const {
    return batch_stats_sorted_unique(wave_index, order_indices).cost;
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
    const std::vector<int>& items = order_items_by_snake_rank_.at(static_cast<std::size_t>(order_index));
    if (items.empty()) {
        return 0;
    }

    const int min_rank = instance_.snake_rank.at(static_cast<std::size_t>(items.front()));
    const int max_rank = instance_.snake_rank.at(static_cast<std::size_t>(items.back()));
    return max_rank - min_rank;
}

std::vector<std::string> BatchEvaluator::validate_solution(const Solution& solution) const {
    std::vector<std::string> errors;
    std::vector<int> assignments(instance_.orders.size(), 0);

    // Controleer eerst elke batch lokaal: bestaande wave, juiste wave-id,
    // capaciteit en alleen orders die bij die wave horen.
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

    // Daarna de globale constraint: elke order moet exact een keer voorkomen.
    for (std::size_t order_index = 0; order_index < assignments.size(); ++order_index) {
        if (assignments[order_index] != 1) {
            errors.push_back(
                "Order " + instance_.orders[order_index].id + " assigned " + std::to_string(assignments[order_index]) + " times");
        }
    }

    return errors;
}

}  // namespace optioryx
