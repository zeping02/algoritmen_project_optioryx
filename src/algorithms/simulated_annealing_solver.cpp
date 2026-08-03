#include "optioryx/algorithms/simulated_annealing_solver.hpp"

#include <algorithm>
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

double evaluate_batch_list(const BatchEvaluator& evaluator, int wave_index, const BatchList& batch_list) {
    // Objective van een tijdelijke SA-oplossing zonder eerst Batch-objecten te bouwen.
    double objective = 0.0;
    for (const std::vector<int>& batch : batch_list) {
        objective += evaluator.batch_cost(wave_index, batch);
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

bool apply_relocate_move(const ProblemInstance& instance, BatchList& batch_list, std::mt19937_64& rng) {
    // Relocate verplaatst een order naar een andere bestaande batch of, als zinvol,
    // naar een nieuwe batch. Capaciteit wordt hierbij behouden.
    if (batch_list.empty()) {
        return false;
    }

    std::vector<std::size_t> source_candidates;
    for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
        if (!batch_list[batch_index].empty()) {
            source_candidates.push_back(batch_index);
        }
    }
    if (source_candidates.empty()) {
        return false;
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
        return false;
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
    if (destination == batch_list.size()) {
        batch_list.push_back({moved_order});
    } else {
        batch_list[destination].push_back(moved_order);
    }

    if (batch_list[source_index].empty()) {
        batch_list.erase(batch_list.begin() + static_cast<std::ptrdiff_t>(source_index));
    }

    return true;
}

bool apply_swap_move(BatchList& batch_list, std::mt19937_64& rng) {
    // Swap wisselt twee orders tussen twee niet-lege batches.
    if (batch_list.size() < 2U) {
        return false;
    }

    std::vector<std::size_t> non_empty_batches;
    for (std::size_t batch_index = 0; batch_index < batch_list.size(); ++batch_index) {
        if (!batch_list[batch_index].empty()) {
            non_empty_batches.push_back(batch_index);
        }
    }
    if (non_empty_batches.size() < 2U) {
        return false;
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
    return true;
}

bool apply_merge_split_move(const ProblemInstance& instance, BatchList& batch_list, std::mt19937_64& rng) {
    // Merge-split combineert twee batches en splitst ze opnieuw.
    // Dit is een grotere move dan relocate/swap en kan lokale structuren doorbreken.
    if (batch_list.size() < 2U) {
        return false;
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
        return false;
    }
    std::shuffle(combined_orders.begin(), combined_orders.end(), rng);

    if (combined_orders.size() <= instance.nmax) {
        batch_list[first_batch] = std::move(combined_orders);
        batch_list.erase(batch_list.begin() + static_cast<std::ptrdiff_t>(second_batch));
        return true;
    }

    const std::size_t min_first_size = combined_orders.size() - instance.nmax;
    const std::size_t max_first_size = std::min(instance.nmax, combined_orders.size() - 1U);
    if (min_first_size > max_first_size) {
        return false;
    }

    std::uniform_int_distribution<std::size_t> pick_split(min_first_size, max_first_size);
    const std::size_t first_size = pick_split(rng);
    batch_list[first_batch].assign(combined_orders.begin(), combined_orders.begin() + static_cast<std::ptrdiff_t>(first_size));
    batch_list[second_batch].assign(combined_orders.begin() + static_cast<std::ptrdiff_t>(first_size), combined_orders.end());
    return true;
}

bool apply_neighbor_move(
    const ProblemInstance& instance,
    NeighborhoodStrategy neighborhood,
    BatchList& batch_list,
    std::mt19937_64& rng) {
    // Bij "mixed" proberen we verschillende move-types. Als de willekeurig gekozen
    // move niet toepasbaar is, vallen we terug op een andere haalbare move.
    if (neighborhood == NeighborhoodStrategy::relocate) {
        return apply_relocate_move(instance, batch_list, rng);
    }
    if (neighborhood == NeighborhoodStrategy::swap) {
        return apply_swap_move(batch_list, rng);
    }
    if (neighborhood == NeighborhoodStrategy::merge_split) {
        return apply_merge_split_move(instance, batch_list, rng);
    }

    std::uniform_int_distribution<int> pick(0, 2);
    const int selected = pick(rng);
    if (selected == 0 && apply_relocate_move(instance, batch_list, rng)) {
        return true;
    }
    if (selected == 1 && apply_swap_move(batch_list, rng)) {
        return true;
    }
    if (selected == 2 && apply_merge_split_move(instance, batch_list, rng)) {
        return true;
    }
    if (apply_relocate_move(instance, batch_list, rng)) {
        return true;
    }
    if (apply_swap_move(batch_list, rng)) {
        return true;
    }
    return apply_merge_split_move(instance, batch_list, rng);
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
        current_batches = to_batch_list(solve_wave_greedy(instance, evaluator, wave_index, GreedyConfig{}).batches);
    } else {
        current_batches = build_randomized_start(instance, wave_index, rng);
    }

    double current_objective = evaluate_batch_list(evaluator, wave_index, current_batches);
    BatchList best_batches = current_batches;
    double best_objective = current_objective;

    double temperature = config.initial_temperature;
    std::uniform_real_distribution<double> probability(0.0, 1.0);

    WaveSolveResult result;
    result.metadata["start_strategy"] = to_string(config.start_strategy);
    result.metadata["neighborhood"] = to_string(config.neighborhood);
    result.metadata["cooling"] = to_string(config.cooling);
    result.metadata["config_id"] = config.config_id;

    result.convergence.push_back({0, best_objective, current_objective, temperature});
    for (int iteration = 1; iteration <= config.iterations; ++iteration) {
        BatchList candidate_batches = current_batches;
        if (!apply_neighbor_move(instance, config.neighborhood, candidate_batches, rng)) {
            result.convergence.push_back({iteration, best_objective, current_objective, temperature});
            temperature = update_temperature(temperature, config);
            continue;
        }

        const double candidate_objective = evaluate_batch_list(evaluator, wave_index, candidate_batches);
        const double delta = candidate_objective - current_objective;
        // Metropolis-acceptatie: slechtere oplossingen kunnen tijdelijk worden aanvaard
        // om uit lokale optima te ontsnappen.
        const bool accept = delta <= 0.0 || probability(rng) < std::exp(-delta / std::max(temperature, 1e-9));
        if (accept) {
            current_batches = std::move(candidate_batches);
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
