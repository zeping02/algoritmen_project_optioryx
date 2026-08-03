#include "optioryx/algorithms/simulated_annealing_solver.hpp"

#include "optioryx/util/format.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "optioryx/algorithms/greedy_solver.hpp"

namespace optioryx {

namespace {

// Simulated annealing werkt op een eenvoudigere representatie:
// een vector van batches, waarbij elke batch een vector orderindices is.
using BatchList = std::vector<std::vector<int>>;

// Een move wijzigt hoogstens twee batches. We bewaren daarom de som van de
// oude kosten en de indices die na de move opnieuw moeten worden geevalueerd.
// De indices verwijzen naar de batchlijst nadat eventuele insert/erase-acties
// zijn uitgevoerd.
struct NeighborMoveResult {
    bool applied = false;
    double previous_affected_cost = 0.0;
    std::array<std::size_t, 2> changed_batch_indices{};
    std::size_t changed_batch_count = 0U;
    std::size_t erased_batch_index = std::numeric_limits<std::size_t>::max();
    bool appended_batch = false;
};

NeighborMoveResult one_changed_batch(double previous_cost, std::size_t batch_index) {
    return {true, previous_cost, {batch_index, 0U}, 1U};
}

NeighborMoveResult two_changed_batches(
    double previous_cost,
    std::size_t first_batch_index,
    std::size_t second_batch_index) {
    return {true, previous_cost, {first_batch_index, second_batch_index}, 2U};
}

BatchList to_batch_list(const std::vector<Batch>& batches) {
    BatchList batch_list;
    batch_list.reserve(batches.size());
    for (const Batch& batch : batches) {
        batch_list.push_back(batch.order_indices);
    }
    return batch_list;
}

std::vector<Batch> to_batches(const std::string& wave_id, const BatchList& batch_list) {
    std::vector<Batch> batches;
    batches.reserve(batch_list.size());
    for (const std::vector<int>& order_indices : batch_list) {
        Batch batch;
        batch.wave_id = wave_id;
        batch.order_indices = order_indices;
        batches.push_back(batch);
    }
    return batches;
}

double evaluate_batch_list(
    const BatchEvaluator& evaluator,
    int wave_index,
    const BatchList& batch_list,
    std::vector<double>* batch_costs = nullptr) {
    // Objective van een tijdelijke SA-oplossing zonder eerst Batch-objecten te bouwen.
    double objective = 0.0;
    if (batch_costs != nullptr) {
        batch_costs->clear();
        batch_costs->reserve(batch_list.size());
    }
    for (const std::vector<int>& batch : batch_list) {
        const double cost = evaluator.batch_cost(wave_index, batch);
        objective += cost;
        if (batch_costs != nullptr) {
            batch_costs->push_back(cost);
        }
    }
    return objective;
}

BatchList build_randomized_start(const ProblemInstance& instance, int wave_index, std::mt19937_64& rng) {
    // Randomized start verdeelt orders willekeurig over batches met capaciteit.
    // Dit werd gebruikt om te vergelijken met een sterke greedy warm-start.
    std::vector<int> shuffled_orders = instance.waves.at(static_cast<std::size_t>(wave_index)).order_indices;
    std::shuffle(shuffled_orders.begin(), shuffled_orders.end(), rng);

    BatchList batch_list;
    for (const int order_index : shuffled_orders) {
        std::vector<std::size_t> eligible_batches;
        for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
            if (batch_list[batch_index].size() < instance.nmax) {
                eligible_batches.push_back(batch_index);
            }
        }

        std::uniform_real_distribution<double> chance(0.0, 1.0);
        const bool create_new_batch = eligible_batches.empty() || chance(rng) < 0.45;
        if (create_new_batch) {
            batch_list.push_back({order_index});
            continue;
        }

        std::uniform_int_distribution<std::size_t> pick_batch(0, eligible_batches.size() - 1U);
        batch_list[eligible_batches[pick_batch(rng)]].push_back(order_index);
    }

    return batch_list;
}

NeighborMoveResult apply_relocate_move(
    const ProblemInstance& instance,
    BatchList& batch_list,
    const std::vector<double>& batch_costs,
    std::mt19937_64& rng) {
    // Relocate verplaatst een order naar een andere bestaande batch of, als zinvol,
    // naar een nieuwe batch. Capaciteit wordt hierbij behouden.
    if (batch_list.empty()) {
        return {};
    }

    std::vector<std::size_t> source_candidates;
    for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
        if (!batch_list[batch_index].empty()) {
            source_candidates.push_back(batch_index);
        }
    }
    if (source_candidates.empty()) {
        return {};
    }

    std::uniform_int_distribution<std::size_t> pick_source(0, source_candidates.size() - 1U);
    const std::size_t source_index = source_candidates[pick_source(rng)];

    std::vector<std::size_t> destination_candidates;
    for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
        if (batch_index != source_index && batch_list[batch_index].size() < instance.nmax) {
            destination_candidates.push_back(batch_index);
        }
    }

    const bool can_create_new = batch_list[source_index].size() > 1U;
    if (destination_candidates.empty() && !can_create_new) {
        return {};
    }

    std::uniform_int_distribution<std::size_t> pick_order(0, batch_list[source_index].size() - 1U);
    const std::size_t order_pos = pick_order(rng);
    const int moved_order = batch_list[source_index][order_pos];
    batch_list[source_index].erase(batch_list[source_index].begin() + static_cast<std::ptrdiff_t>(order_pos));

    std::vector<std::size_t> all_destinations = destination_candidates;
    if (can_create_new) {
        all_destinations.push_back(batch_list.size());
    }

    std::uniform_int_distribution<std::size_t> pick_destination(0, all_destinations.size() - 1U);
    const std::size_t destination = all_destinations[pick_destination(rng)];
    NeighborMoveResult result;
    result.applied = true;
    result.previous_affected_cost = batch_costs[source_index];
    if (destination == batch_list.size()) {
        batch_list.push_back({moved_order});
        result.appended_batch = true;
    } else {
        result.previous_affected_cost += batch_costs[destination];
        batch_list[destination].push_back(moved_order);
    }

    if (batch_list[source_index].empty()) {
        batch_list.erase(batch_list.begin() + static_cast<std::ptrdiff_t>(source_index));
        result.erased_batch_index = source_index;
        // Een lege bron kan alleen naar een bestaande batch verhuizen. Door het
        // wissen schuift de bestemming eventueel een positie naar links.
        result.changed_batch_indices[0] = destination > source_index ? destination - 1U : destination;
        result.changed_batch_count = 1U;
    } else {
        result.changed_batch_indices = {source_index, destination};
        result.changed_batch_count = 2U;
    }

    return result;
}

NeighborMoveResult apply_swap_move(
    BatchList& batch_list,
    const std::vector<double>& batch_costs,
    std::mt19937_64& rng) {
    // Swap wisselt twee orders tussen twee niet-lege batches.
    if (batch_list.size() < 2U) {
        return {};
    }

    std::vector<std::size_t> non_empty_batches;
    for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
        if (!batch_list[batch_index].empty()) {
            non_empty_batches.push_back(batch_index);
        }
    }
    if (non_empty_batches.size() < 2U) {
        return {};
    }

    std::uniform_int_distribution<std::size_t> pick_batch(0, non_empty_batches.size() - 1U);
    const std::size_t first_batch = non_empty_batches[pick_batch(rng)];
    std::size_t second_batch = first_batch;
    while (second_batch == first_batch) {
        second_batch = non_empty_batches[pick_batch(rng)];
    }

    std::uniform_int_distribution<std::size_t> pick_first_order(0, batch_list[first_batch].size() - 1U);
    std::uniform_int_distribution<std::size_t> pick_second_order(0, batch_list[second_batch].size() - 1U);
    const std::size_t first_pos = pick_first_order(rng);
    const std::size_t second_pos = pick_second_order(rng);
    std::swap(batch_list[first_batch][first_pos], batch_list[second_batch][second_pos]);
    return two_changed_batches(
        batch_costs[first_batch] + batch_costs[second_batch],
        first_batch,
        second_batch);
}

NeighborMoveResult apply_merge_split_move(
    const ProblemInstance& instance,
    BatchList& batch_list,
    const std::vector<double>& batch_costs,
    std::mt19937_64& rng) {
    // Merge-split combineert twee batches en splitst ze opnieuw.
    // Dit is een grotere move dan relocate/swap en kan lokale structuren doorbreken.
    if (batch_list.size() < 2U) {
        return {};
    }

    std::uniform_int_distribution<std::size_t> pick_batch(0, batch_list.size() - 1U);
    std::size_t first_batch = pick_batch(rng);
    std::size_t second_batch = first_batch;
    while (second_batch == first_batch) {
        second_batch = pick_batch(rng);
    }
    if (first_batch > second_batch) {
        std::swap(first_batch, second_batch);
    }

    std::vector<int> combined_orders = batch_list[first_batch];
    combined_orders.insert(combined_orders.end(), batch_list[second_batch].begin(), batch_list[second_batch].end());
    if (combined_orders.size() < 2U) {
        return {};
    }
    std::shuffle(combined_orders.begin(), combined_orders.end(), rng);

    const double previous_affected_cost = batch_costs[first_batch] + batch_costs[second_batch];

    if (combined_orders.size() <= instance.nmax) {
        batch_list[first_batch] = std::move(combined_orders);
        batch_list.erase(batch_list.begin() + static_cast<std::ptrdiff_t>(second_batch));
        NeighborMoveResult result = one_changed_batch(previous_affected_cost, first_batch);
        result.erased_batch_index = second_batch;
        return result;
    }

    const std::size_t min_first_size = combined_orders.size() - instance.nmax;
    const std::size_t max_first_size = std::min(instance.nmax, combined_orders.size() - 1U);
    if (min_first_size > max_first_size) {
        return {};
    }

    std::uniform_int_distribution<std::size_t> pick_split(min_first_size, max_first_size);
    const std::size_t first_size = pick_split(rng);
    batch_list[first_batch].assign(combined_orders.begin(), combined_orders.begin() + static_cast<std::ptrdiff_t>(first_size));
    batch_list[second_batch].assign(combined_orders.begin() + static_cast<std::ptrdiff_t>(first_size), combined_orders.end());
    return two_changed_batches(previous_affected_cost, first_batch, second_batch);
}

NeighborMoveResult apply_neighbor_move(
    const ProblemInstance& instance,
    NeighborhoodStrategy neighborhood,
    BatchList& batch_list,
    const std::vector<double>& batch_costs,
    std::mt19937_64& rng) {
    // Bij "mixed" proberen we verschillende move-types. Als de willekeurig gekozen
    // move niet toepasbaar is, vallen we terug op een andere haalbare move.
    if (neighborhood == NeighborhoodStrategy::relocate) {
        return apply_relocate_move(instance, batch_list, batch_costs, rng);
    }
    if (neighborhood == NeighborhoodStrategy::swap) {
        return apply_swap_move(batch_list, batch_costs, rng);
    }
    if (neighborhood == NeighborhoodStrategy::merge_split) {
        return apply_merge_split_move(instance, batch_list, batch_costs, rng);
    }

    std::uniform_int_distribution<int> pick(0, 2);
    const int selected = pick(rng);
    NeighborMoveResult result;
    if (selected == 0 && (result = apply_relocate_move(instance, batch_list, batch_costs, rng)).applied) {
        return result;
    }
    if (selected == 1 && (result = apply_swap_move(batch_list, batch_costs, rng)).applied) {
        return result;
    }
    if (selected == 2 && (result = apply_merge_split_move(instance, batch_list, batch_costs, rng)).applied) {
        return result;
    }
    if ((result = apply_relocate_move(instance, batch_list, batch_costs, rng)).applied) {
        return result;
    }
    if ((result = apply_swap_move(batch_list, batch_costs, rng)).applied) {
        return result;
    }
    return apply_merge_split_move(instance, batch_list, batch_costs, rng);
}

double update_temperature(double temperature, const SimulatedAnnealingConfig& config) {
    // Temperatuurdaling bepaalt hoe vaak slechtere oplossingen later nog geaccepteerd worden.
    if (config.cooling == CoolingSchedule::geometric) {
        return std::max(config.final_temperature, temperature * config.geometric_alpha);
    }
    const double updated = temperature / (1.0 + config.lundy_beta * temperature);
    return std::max(config.final_temperature, updated);
}

}  // namespace

WaveSolveResult solve_wave_simulated_annealing(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const SimulatedAnnealingConfig& config,
    std::mt19937_64& rng) {
    // SA accepteert altijd verbeteringen en soms verslechteringen.
    // Die kans neemt af met de temperatuur, zodat het algoritme eerst exploreert
    // en later meer op lokale verbetering lijkt.
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    BatchList current_batches;
    if (config.start_strategy == StartSolutionStrategy::greedy) {
        GreedyConfig start_config;
        start_config.config_id =
            "sa_greedy_start_ls-" + std::to_string(config.greedy_start_local_search_passes) +
            "_win-" + std::to_string(config.greedy_start_local_search_neighbor_window);
        start_config.local_search_passes = config.greedy_start_local_search_passes;
        start_config.local_search_neighbor_window = config.greedy_start_local_search_neighbor_window;
        current_batches = to_batch_list(solve_wave_greedy(instance, evaluator, wave_index, start_config).batches);
    } else {
        current_batches = build_randomized_start(instance, wave_index, rng);
    }

    std::vector<double> current_batch_costs;
    double current_objective = evaluate_batch_list(evaluator, wave_index, current_batches, &current_batch_costs);
    BatchList best_batches = current_batches;
    double best_objective = current_objective;

    double temperature = config.initial_temperature;
    std::uniform_real_distribution<double> probability(0.0, 1.0);

    WaveSolveResult result;
    result.metadata["start_strategy"] = to_string(config.start_strategy);
    result.metadata["greedy_start_local_search_passes"] =
        std::to_string(config.greedy_start_local_search_passes);
    result.metadata["greedy_start_local_search_neighbor_window"] =
        std::to_string(config.greedy_start_local_search_neighbor_window);
    result.metadata["neighborhood"] = to_string(config.neighborhood);
    result.metadata["cooling"] = to_string(config.cooling);
    result.metadata["iterations"] = std::to_string(config.iterations);
    result.metadata["initial_temperature"] = format_double(config.initial_temperature);
    result.metadata["final_temperature"] = format_double(config.final_temperature);
    result.metadata["geometric_alpha"] = format_double(config.geometric_alpha);
    result.metadata["lundy_beta"] = format_double(config.lundy_beta);
    result.metadata["config_id"] = config.config_id;

    result.convergence.push_back({0, best_objective, current_objective, temperature});
    for (int iteration = 1; iteration <= config.iterations; ++iteration) {
        BatchList candidate_batches = current_batches;
        const NeighborMoveResult move =
            apply_neighbor_move(instance, config.neighborhood, candidate_batches, current_batch_costs, rng);
        if (!move.applied) {
            result.convergence.push_back({iteration, best_objective, current_objective, temperature});
            temperature = update_temperature(temperature, config);
            continue;
        }

        // Alleen de een of twee batches die werkelijk veranderden worden opnieuw
        // gescoord. Alle ongewijzigde bijdragen vallen exact weg in de delta.
        double new_affected_cost = 0.0;
        std::array<double, 2> changed_batch_costs{};
        for (std::size_t changed_index = 0U; changed_index < move.changed_batch_count; ++changed_index) {
            const std::size_t batch_index = move.changed_batch_indices[changed_index];
            const double cost = evaluator.batch_cost(wave_index, candidate_batches[batch_index]);
            changed_batch_costs[changed_index] = cost;
            new_affected_cost += cost;
        }
        const double delta = new_affected_cost - move.previous_affected_cost;
        const double candidate_objective = current_objective + delta;
        // Metropolis-acceptatie: slechtere oplossingen kunnen tijdelijk worden aanvaard
        // om uit lokale optima te ontsnappen.
        const bool accept = delta <= 0.0 || probability(rng) < std::exp(-delta / std::max(temperature, 1e-9));
        if (accept) {
            current_batches = std::move(candidate_batches);
            if (move.erased_batch_index != std::numeric_limits<std::size_t>::max()) {
                current_batch_costs.erase(
                    current_batch_costs.begin() + static_cast<std::ptrdiff_t>(move.erased_batch_index));
            }
            if (move.appended_batch) {
                current_batch_costs.push_back(0.0);
            }
            for (std::size_t changed_index = 0U; changed_index < move.changed_batch_count; ++changed_index) {
                current_batch_costs[move.changed_batch_indices[changed_index]] = changed_batch_costs[changed_index];
            }
            current_objective = candidate_objective;
        }
        if (current_objective < best_objective) {
            best_objective = current_objective;
            best_batches = current_batches;
        }

        result.convergence.push_back({iteration, best_objective, current_objective, temperature});
        temperature = update_temperature(temperature, config);
    }

    result.batches = to_batches(wave.id, best_batches);
    result.objective = best_objective;

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
