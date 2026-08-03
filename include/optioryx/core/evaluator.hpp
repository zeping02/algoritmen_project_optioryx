#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
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

// Transparante string-hash: een cache-hit kan rechtstreeks op de bytes van de
// reeds gesorteerde int-vector zoeken, zonder eerst een tijdelijke string te
// alloceren. Alleen cache-misses materialiseren een owning key.
struct BatchCacheKeyHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

struct BatchCacheKeyEqual {
    using is_transparent = void;

    bool operator()(std::string_view left, std::string_view right) const noexcept {
        return left == right;
    }
};

// BatchEvaluator is het centrale scorings- en validatiepunt.
//
// De evaluator bepaalt de echte objective zoals die lokaal gebruikt wordt:
// unieke locaties per batch verzamelen, in snake-volgorde bezoeken en de
// afstanden START -> locaties -> END optellen. Omdat dezelfde batches vaak
// opnieuw getest worden tijdens local search, cachet deze klasse batch_stats.
class BatchEvaluator {
public:
    // De limiet geldt over alle waves samen. Een limiet van 0 schakelt de
    // cache uit; zo blijft ook een lange metaheuristiek binnen een vaste
    // geheugenvoetafdruk.
    explicit BatchEvaluator(const ProblemInstance& instance, std::size_t cache_entry_limit = 100'000U);

    // Geeft alle routekenmerken terug voor een set orders binnen een wave.
    BatchStats batch_stats(int wave_index, const std::vector<int>& order_indices) const;
    // Snelle variant voor callers die de orderindices al strikt oplopend en
    // zonder dubbels aanleveren. De GA-decoder gebruikt dit in zijn hot path.
    BatchStats batch_stats_sorted_unique(int wave_index, const std::vector<int>& order_indices) const;
    // Alleen de kostcomponent van batch_stats.
    double batch_cost(int wave_index, const std::vector<int>& order_indices) const;
    double batch_cost_sorted_unique(int wave_index, const std::vector<int>& order_indices) const;
    // Totale kost van alle batches binnen een wave.
    double wave_cost(int wave_index, const std::vector<Batch>& batches) const;
    // Totale kost over alle waves heen.
    double solution_cost(const Solution& solution) const;
    // Controleert harde constraints: order exact een keer, juiste wave, nmax.
    std::vector<std::string> validate_solution(const Solution& solution) const;

    // Breedte tussen de eerste en laatste locatie van een order in snake-volgorde.
    int order_span_width(int order_index) const;

    // Aantal momenteel gecachete batches over alle waves samen.
    // Vooral nuttig om de geheugenlimiet in regressietests te bewaken.
    std::size_t cache_size() const noexcept;

private:
    // De private helpers verwachten strikt oplopende, unieke orderindices.
    std::string_view make_key_view(const std::vector<int>& sorted_unique_order_indices) const;
    BatchStats compute_stats_sorted_unique(
        int wave_index,
        const std::vector<int>& sorted_unique_order_indices) const;

    const ProblemInstance& instance_;
    std::size_t mask_words_ = 0;
    std::vector<int> items_by_snake_rank_;
    std::vector<std::vector<std::uint64_t>> order_item_masks_;
    std::vector<std::vector<int>> order_items_by_snake_rank_;
    mutable std::vector<std::uint64_t> item_mask_scratch_;
    std::size_t cache_entry_limit_ = 0;
    mutable std::size_t cache_size_ = 0;
    using WaveCache = std::unordered_map<std::string, BatchStats, BatchCacheKeyHash, BatchCacheKeyEqual>;
    mutable std::vector<WaveCache> cache_;
};

}  // namespace optioryx
