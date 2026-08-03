#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "optioryx/core/model.hpp"

namespace optioryx {

// Samenvatting van een batch/picklijst.
// De solver gebruikt niet alleen de kost, maar ook snake-span en item-count
// om routes te vergelijken en lokale verbeteringen te sturen.
struct BatchStats {
    double cost = 0.0;
    int min_snake_rank = 0;
    int max_snake_rank = 0;
    int span_width = 0;
    std::size_t unique_item_count = 0;
};

// BatchEvaluator is het centrale scorings- en validatiepunt.
//
// De evaluator bepaalt de echte objective zoals die lokaal gebruikt wordt:
// unieke locaties per batch verzamelen, in snake-volgorde bezoeken en de
// afstanden START -> locaties -> END optellen. Omdat dezelfde batches vaak
// opnieuw getest worden tijdens local search, cachet deze klasse batch_stats.
class BatchEvaluator {
public:
    explicit BatchEvaluator(const ProblemInstance& instance);

    // Geeft alle routekenmerken terug voor een set orders binnen een wave.
    BatchStats batch_stats(int wave_index, const std::vector<int>& order_indices) const;
    // Alleen de kostcomponent van batch_stats.
    double batch_cost(int wave_index, const std::vector<int>& order_indices) const;
    // Totale kost van alle batches binnen een wave.
    double wave_cost(int wave_index, const std::vector<Batch>& batches) const;
    // Totale kost over alle waves heen.
    double solution_cost(const Solution& solution) const;
    // Controleert harde constraints: order exact een keer, juiste wave, nmax.
    std::vector<std::string> validate_solution(const Solution& solution) const;

    // Breedte tussen de eerste en laatste locatie van een order in snake-volgorde.
    int order_span_width(int order_index) const;

private:
    // Canonieke cache-key: dezelfde orderverzameling krijgt dezelfde key,
    // onafhankelijk van de volgorde waarin orders worden aangeleverd.
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
