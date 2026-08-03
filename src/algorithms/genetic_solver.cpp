#include "optioryx/algorithms/genetic_solver.hpp"

#include "optioryx/util/format.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "optioryx/algorithms/greedy_solver.hpp"

namespace optioryx {

namespace {

// Een chromosoom is een permutatie van orders.
// De decoder zet die permutatie achteraf om naar geldige batches, zodat crossover
// en mutation nooit rechtstreeks met LIST_ID's of capaciteit hoeven te werken.
struct Chromosome {
    std::vector<int> permutation;
    double objective = 0.0;
};

// Deze buffers worden voor alle fitnessberekeningen binnen een wave hergebruikt.
// De gewone publieke decoder bouwt ook de uiteindelijke batches op; tijdens de
// evolutie heeft een chromosoom alleen zijn objective nodig.
struct DecodeWorkspace {
    std::vector<double> dp;
    std::vector<int> batch_orders;
};

// Generatiestempels maken membershiptests bij order crossover O(1), zonder de
// markering voor elke nieuwe child volledig te moeten wissen.
struct CrossoverWorkspace {
    explicit CrossoverWorkspace(const std::size_t order_count) : marks(order_count, 0U) {}

    std::uint32_t next_stamp() {
        if (stamp == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(marks.begin(), marks.end(), 0U);
            stamp = 1U;
        } else {
            ++stamp;
        }
        return stamp;
    }

    std::vector<std::uint32_t> marks;
    std::uint32_t stamp = 0U;
};

// Gebruik de greedy-oplossing als sterke warm-start in de populatie.
// De overige individuen zijn shuffles, zodat er nog diversiteit is.
std::vector<int> greedy_seed_permutation(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GeneticConfig& config) {
    GreedyConfig start_config;
    start_config.local_search_passes = config.greedy_start_local_search_passes;
    start_config.local_search_neighbor_window = config.greedy_start_local_search_neighbor_window;
    const WaveSolveResult greedy = solve_wave_greedy(instance, evaluator, wave_index, start_config);
    std::vector<int> permutation;
    for (const Batch& batch : greedy.batches) {
        permutation.insert(permutation.end(), batch.order_indices.begin(), batch.order_indices.end());
    }
    return permutation;
}

double decode_permutation_objective(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& permutation,
    DecodeWorkspace& workspace) {
    const std::size_t n = permutation.size();
    const double infinity = std::numeric_limits<double>::infinity();
    workspace.dp.assign(n + 1U, infinity);
    workspace.dp[0] = 0.0;
    workspace.batch_orders.reserve(instance.nmax);

    for (std::size_t end = 1; end <= n; ++end) {
        const std::size_t max_batch_size = std::min<std::size_t>(instance.nmax, end);
        workspace.batch_orders.clear();
        for (std::size_t batch_size = 1; batch_size <= max_batch_size; ++batch_size) {
            const std::size_t start = end - batch_size;
            const int added_order = permutation[start];
            workspace.batch_orders.insert(
                std::lower_bound(workspace.batch_orders.begin(), workspace.batch_orders.end(), added_order),
                added_order);
            const double candidate =
                workspace.dp[start] + evaluator.batch_cost_sorted_unique(wave_index, workspace.batch_orders);
            if (candidate < workspace.dp[end]) {
                workspace.dp[end] = candidate;
            }
        }
    }
    if (n > 0U && workspace.dp[n] == infinity) {
        throw std::runtime_error("Failed to decode permutation into batches");
    }
    return workspace.dp[n];
}

Chromosome make_chromosome(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    std::vector<int> permutation,
    DecodeWorkspace& workspace) {
    // Elke nieuwe permutatie wordt meteen gescoord; batchconstructie gebeurt
    // eenmaal voor de uiteindelijke winnaar.
    Chromosome chromosome;
    chromosome.objective = decode_permutation_objective(instance, evaluator, wave_index, permutation, workspace);
    chromosome.permutation = std::move(permutation);
    return chromosome;
}

int tournament_pick(const std::vector<Chromosome>& population, int tournament_size, std::mt19937_64& rng) {
    // Tournament selection kiest meerdere willekeurige kandidaten en neemt de beste.
    // Dit geeft selectiedruk zonder dat alleen de absolute top mag reproduceren.
    std::uniform_int_distribution<int> pick(0, static_cast<int>(population.size() - 1U));
    int best_index = pick(rng);
    for (int draw = 1; draw < tournament_size; ++draw) {
        const int contender = pick(rng);
        if (population[static_cast<std::size_t>(contender)].objective <
            population[static_cast<std::size_t>(best_index)].objective) {
            best_index = contender;
        }
    }
    return best_index;
}

std::vector<int> order_crossover(
    const std::vector<int>& left,
    const std::vector<int>& right,
    std::mt19937_64& rng,
    CrossoverWorkspace& workspace) {
    // Order crossover bewaart een segment van de eerste ouder en vult de rest aan
    // volgens de volgorde van de tweede ouder. Zo blijft elke order exact een keer aanwezig.
    if (left.empty()) {
        return {};
    }
    if (left.size() == 1U) {
        return {left.front()};
    }

    std::uniform_int_distribution<std::size_t> pick(0, left.size() - 1U);
    std::size_t first = pick(rng);
    std::size_t second = pick(rng);
    if (first > second) {
        std::swap(first, second);
    }

    const std::uint32_t stamp = workspace.next_stamp();
    std::vector<int> child(left.size(), -1);
    for (std::size_t index = first; index <= second; ++index) {
        child[index] = left[index];
        workspace.marks.at(static_cast<std::size_t>(left[index])) = stamp;
    }

    std::size_t insert_pos = (second + 1U) % left.size();
    for (std::size_t offset = 0; offset < right.size(); ++offset) {
        const int candidate = right[(second + 1U + offset) % right.size()];
        if (workspace.marks.at(static_cast<std::size_t>(candidate)) == stamp) {
            continue;
        }
        child[insert_pos] = candidate;
        workspace.marks[static_cast<std::size_t>(candidate)] = stamp;
        insert_pos = (insert_pos + 1U) % child.size();
    }

    return child;
}

void mutate_permutation(std::vector<int>& permutation, double insertion_rate, std::mt19937_64& rng) {
    // Mutatie wisselt twee orders of verplaatst een order naar een andere positie.
    // Insertion is vaak nuttiger dan pure swap omdat batching sterk afhangt van lokale volgorde.
    if (permutation.size() < 2U) {
        return;
    }

    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, permutation.size() - 1U);
    std::size_t first = pick(rng);
    std::size_t second = pick(rng);
    while (second == first) {
        second = pick(rng);
    }

    if (probability(rng) < insertion_rate) {
        const int value = permutation[first];
        permutation.erase(permutation.begin() + static_cast<std::ptrdiff_t>(first));
        if (first < second) {
            --second;
        }
        permutation.insert(permutation.begin() + static_cast<std::ptrdiff_t>(second), value);
        return;
    }

    std::swap(permutation[first], permutation[second]);
}

double population_diversity(const std::vector<Chromosome>& population) {
    // Eenvoudige diversiteitsmaat: gemiddelde fractie posities waarin twee permutaties verschillen.
    // Wordt alleen voor analyse/convergentie gebruikt.
    if (population.size() < 2U || population.front().permutation.empty()) {
        return 0.0;
    }

    double total = 0.0;
    int pair_count = 0;
    for (std::size_t left = 0; left < population.size(); ++left) {
        for (std::size_t right = left + 1U; right < population.size(); ++right) {
            int mismatches = 0;
            for (std::size_t index = 0; index < population[left].permutation.size(); ++index) {
                if (population[left].permutation[index] != population[right].permutation[index]) {
                    ++mismatches;
                }
            }
            total += static_cast<double>(mismatches) /
                     static_cast<double>(population[left].permutation.size());
            ++pair_count;
        }
    }

    return pair_count == 0 ? 0.0 : total / static_cast<double>(pair_count);
}

}  // namespace

DecodeResult decode_permutation_to_batches(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& permutation) {
    // Dynamic programming decoder:
    // dp[end] = laagste kost om de eerste end orders uit de permutatie in batches te splitsen.
    // Elke batch is een aaneengesloten segment van de permutatie en heeft maximaal nmax orders.
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));
    const std::size_t n = permutation.size();
    const double infinity = std::numeric_limits<double>::infinity();

    std::vector<double> dp(n + 1U, infinity);
    std::vector<int> parent(n + 1U, -1);
    dp[0] = 0.0;

    for (std::size_t end = 1; end <= n; ++end) {
        const std::size_t max_batch_size = std::min<std::size_t>(instance.nmax, end);
        for (std::size_t batch_size = 1; batch_size <= max_batch_size; ++batch_size) {
            const std::size_t start = end - batch_size;
            std::vector<int> batch_orders(
                permutation.begin() + static_cast<std::ptrdiff_t>(start),
                permutation.begin() + static_cast<std::ptrdiff_t>(end));
            const double candidate = dp[start] + evaluator.batch_cost(wave_index, batch_orders);
            if (candidate < dp[end]) {
                dp[end] = candidate;
                parent[end] = static_cast<int>(start);
            }
        }
    }

    std::vector<Batch> batches;
    for (std::size_t end = n; end > 0;) {
        const int start = parent[end];
        if (start < 0) {
            throw std::runtime_error("Failed to decode permutation into batches");
        }
        Batch batch;
        batch.wave_id = wave.id;
        batch.order_indices.assign(
            permutation.begin() + start,
            permutation.begin() + static_cast<std::ptrdiff_t>(end));
        batches.push_back(batch);
        end = static_cast<std::size_t>(start);
    }
    std::reverse(batches.begin(), batches.end());

    DecodeResult result;
    result.batches = std::move(batches);
    result.objective = dp[n];
    return result;
}

WaveSolveResult solve_wave_genetic(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GeneticConfig& config,
    std::mt19937_64& rng) {
    // Genetisch algoritme per wave. Het algoritme zoekt in de ruimte van orderpermutaties;
    // de DP-decoder garandeert daarna een geldige batching voor die permutatie.
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    const int population_size = std::max(2, config.population_size);
    std::vector<Chromosome> population;
    population.reserve(static_cast<std::size_t>(population_size));
    DecodeWorkspace decode_workspace;
    CrossoverWorkspace crossover_workspace(instance.orders.size());

    population.push_back(make_chromosome(
        instance,
        evaluator,
        wave_index,
        greedy_seed_permutation(instance, evaluator, wave_index, config),
        decode_workspace));

    std::vector<int> base_permutation = wave.order_indices;
    while (static_cast<int>(population.size()) < population_size) {
        std::shuffle(base_permutation.begin(), base_permutation.end(), rng);
        population.push_back(make_chromosome(instance, evaluator, wave_index, base_permutation, decode_workspace));
    }

    auto by_objective = [](const Chromosome& left, const Chromosome& right) {
        return left.objective < right.objective;
    };

    WaveSolveResult result;
    result.metadata["greedy_start_local_search_passes"] =
        std::to_string(config.greedy_start_local_search_passes);
    result.metadata["greedy_start_local_search_neighbor_window"] =
        std::to_string(config.greedy_start_local_search_neighbor_window);
    result.metadata["population_size"] = std::to_string(population_size);
    result.metadata["generations"] = std::to_string(config.generations);
    result.metadata["tournament_size"] = std::to_string(std::max(2, config.tournament_size));
    result.metadata["mutation_rate"] = format_double(config.mutation_rate);
    result.metadata["insertion_rate"] = format_double(config.insertion_rate);
    result.metadata["elitism_fraction"] = format_double(config.elitism_fraction);
    result.metadata["representation"] = "order_permutation";
    result.metadata["crossover"] = "order_crossover";
    result.metadata["fitness"] = "decoded_total_route_cost";
    result.metadata["config_id"] = config.config_id;

    Chromosome best = *std::min_element(population.begin(), population.end(), by_objective);
    result.convergence.push_back({0, best.objective, best.objective, population_diversity(population)});

    std::uniform_real_distribution<double> probability(0.0, 1.0);
    const int elitism_count = std::clamp(
        static_cast<int>(std::round(static_cast<double>(population_size) * config.elitism_fraction)),
        0,
        population_size);
    result.metadata["elitism_count"] = std::to_string(elitism_count);

    for (int generation = 1; generation <= config.generations; ++generation) {
        // Elitisme bewaart de beste individuen ongewijzigd. De rest van de nieuwe populatie
        // ontstaat via tournament selection, order crossover en optionele mutatie.
        std::sort(population.begin(), population.end(), by_objective);

        std::vector<Chromosome> next_population;
        next_population.reserve(population.size());
        for (int elite = 0; elite < elitism_count; ++elite) {
            next_population.push_back(population[static_cast<std::size_t>(elite)]);
        }

        while (next_population.size() < population.size()) {
            const int left_index = tournament_pick(population, std::max(2, config.tournament_size), rng);
            const int right_index = tournament_pick(population, std::max(2, config.tournament_size), rng);

            std::vector<int> child_permutation = order_crossover(
                population[static_cast<std::size_t>(left_index)].permutation,
                population[static_cast<std::size_t>(right_index)].permutation,
                rng,
                crossover_workspace);

            if (probability(rng) < config.mutation_rate) {
                mutate_permutation(child_permutation, config.insertion_rate, rng);
            }

            next_population.push_back(
                make_chromosome(instance, evaluator, wave_index, std::move(child_permutation), decode_workspace));
        }

        population = std::move(next_population);
        const auto generation_best = std::min_element(population.begin(), population.end(), by_objective);
        if (generation_best->objective < best.objective) {
            best = *generation_best;
        }
        result.convergence.push_back(
            {generation, best.objective, generation_best->objective, population_diversity(population)});
    }

    DecodeResult decoded_best = decode_permutation_to_batches(instance, evaluator, wave_index, best.permutation);
    result.batches = std::move(decoded_best.batches);
    result.objective = best.objective;
    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
