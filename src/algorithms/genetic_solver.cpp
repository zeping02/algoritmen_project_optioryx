#include "optioryx/algorithms/genetic_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "optioryx/algorithms/greedy_solver.hpp"

namespace optioryx {

namespace {

struct Chromosome {
    std::vector<int> permutation;
    DecodeResult decoded;
};

std::vector<int> greedy_seed_permutation(const ProblemInstance& instance, const BatchEvaluator& evaluator, int wave_index) {
    const WaveSolveResult greedy = solve_wave_greedy(instance, evaluator, wave_index, GreedyConfig{});
    std::vector<int> permutation;
    for (const Batch& batch : greedy.batches) {
        permutation.insert(permutation.end(), batch.order_indices.begin(), batch.order_indices.end());
    }
    return permutation;
}

Chromosome make_chromosome(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& permutation) {
    Chromosome chromosome;
    chromosome.permutation = permutation;
    chromosome.decoded = decode_permutation_to_batches(instance, evaluator, wave_index, chromosome.permutation);
    return chromosome;
}

int tournament_pick(const std::vector<Chromosome>& population, int tournament_size, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(population.size() - 1U));
    int best_index = pick(rng);
    for (int draw = 1; draw < tournament_size; ++draw) {
        const int contender = pick(rng);
        if (population[static_cast<std::size_t>(contender)].decoded.objective <
            population[static_cast<std::size_t>(best_index)].decoded.objective) {
            best_index = contender;
        }
    }
    return best_index;
}

std::vector<int> order_crossover(const std::vector<int>& left, const std::vector<int>& right, std::mt19937_64& rng) {
    if (left.size() < 2U) {
        return left;
    }

    std::uniform_int_distribution<std::size_t> pick(0, left.size() - 1U);
    std::size_t first = pick(rng);
    std::size_t second = pick(rng);
    if (first > second) {
        std::swap(first, second);
    }

    std::vector<int> child(left.size(), -1);
    for (std::size_t index = first; index <= second; ++index) {
        child[index] = left[index];
    }

    std::size_t insert_pos = (second + 1U) % left.size();
    for (std::size_t offset = 0; offset < right.size(); ++offset) {
        const int candidate = right[(second + 1U + offset) % right.size()];
        if (std::find(child.begin(), child.end(), candidate) != child.end()) {
            continue;
        }
        child[insert_pos] = candidate;
        insert_pos = (insert_pos + 1U) % child.size();
    }

    return child;
}

void mutate_permutation(std::vector<int>& permutation, double insertion_rate, std::mt19937_64& rng) {
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
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    const int population_size = std::max(2, config.population_size);
    std::vector<Chromosome> population;
    population.reserve(static_cast<std::size_t>(population_size));

    population.push_back(make_chromosome(instance, evaluator, wave_index, greedy_seed_permutation(instance, evaluator, wave_index)));

    std::vector<int> base_permutation = wave.order_indices;
    while (static_cast<int>(population.size()) < population_size) {
        std::shuffle(base_permutation.begin(), base_permutation.end(), rng);
        population.push_back(make_chromosome(instance, evaluator, wave_index, base_permutation));
    }

    auto by_objective = [](const Chromosome& left, const Chromosome& right) {
        return left.decoded.objective < right.decoded.objective;
    };

    WaveSolveResult result;
    result.metadata["population_size"] = std::to_string(population_size);
    result.metadata["generations"] = std::to_string(config.generations);
    result.metadata["mutation_rate"] = std::to_string(config.mutation_rate);
    result.metadata["config_id"] = config.config_id;

    Chromosome best = *std::min_element(population.begin(), population.end(), by_objective);
    result.convergence.push_back({0, best.decoded.objective, best.decoded.objective, population_diversity(population)});

    std::uniform_real_distribution<double> probability(0.0, 1.0);
    const int elitism_count = std::clamp(
        static_cast<int>(std::round(static_cast<double>(population_size) * config.elitism_fraction)),
        1,
        population_size);

    for (int generation = 1; generation <= config.generations; ++generation) {
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
                rng);

            if (probability(rng) < config.mutation_rate) {
                mutate_permutation(child_permutation, config.insertion_rate, rng);
            }

            next_population.push_back(make_chromosome(instance, evaluator, wave_index, child_permutation));
        }

        population = std::move(next_population);
        const Chromosome generation_best = *std::min_element(population.begin(), population.end(), by_objective);
        if (generation_best.decoded.objective < best.decoded.objective) {
            best = generation_best;
        }
        result.convergence.push_back(
            {generation, best.decoded.objective, generation_best.decoded.objective, population_diversity(population)});
    }

    result.batches = best.decoded.batches;
    result.objective = best.decoded.objective;
    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
