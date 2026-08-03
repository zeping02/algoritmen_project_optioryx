#pragma once

#include <string>

#include "optioryx/algorithms/common.hpp"
#include "optioryx/core/evaluator.hpp"

namespace optioryx {

// Configuratie van de uiteindelijke greedy-familie.
//
// De solver bestaat uit meerdere fasen:
// 1. savings-greedy constructie;
// 2. lokale pairwise swaps/relocaties;
// 3. route-eliminatie om kleine routes op te lossen;
// 4. pair- en triple-rebuilds om route-inhoud opnieuw te verdelen.
//
// Alle parameters staan hier bewust expliciet zodat experimenten reproduceerbaar
// blijven via configbestanden in plaats van via hardcoded codewijzigingen.
struct GreedyConfig {
    std::string config_id = "greedy_default";
    // Aantal passes van eenvoudige lokale zoekacties tussen nabije routes.
    int local_search_passes = 1;
    // Alleen routes met een nabije snake-center worden samen bekeken.
    int local_search_neighbor_window = 8;
    // Route-eliminatie probeert kleine batches volledig in andere batches te plaatsen.
    int route_elimination_passes = 0;
    int route_elimination_max_batch_size = 12;
    int route_elimination_beam_width = 1;
    int route_elimination_candidates_per_order = 1;
    int route_elimination_ordering_trials = 1;
    int post_elimination_local_search_passes = 0;
    // Pair rebuild herverdeelt twee nabije routes volledig als dat de kost verlaagt.
    int pair_rebuild_passes = 0;
    int pair_rebuild_neighbor_window = 0;
    // Triple rebuild doet hetzelfde voor drie routes en kan pair-local optima doorbreken.
    int triple_rebuild_passes = 0;
    int triple_rebuild_neighbor_window = 0;
    // Extra lokale combinaties naast vaste patronen zoals (i, i+2, i+4).
    int triple_rebuild_combination_window = 0;
    // Tail-fasen ruimen verbeteringen op die door eerdere rebuilds ontstaan.
    int post_triple_pair_rebuild_passes = 0;
    int post_triple_pair_rebuild_neighbor_window = 0;
    int post_pair_triple_rebuild_passes = 0;
    int post_pair_triple_rebuild_neighbor_window = 0;
    int post_pair_triple_rebuild_combination_window = 0;
    int post_second_triple_pair_rebuild_passes = 0;
    int post_second_triple_pair_rebuild_neighbor_window = 0;
    int final_local_search_passes = 0;
};

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config);

}  // namespace optioryx
