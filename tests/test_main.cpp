#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "optioryx/algorithms/genetic_solver.hpp"
#include "optioryx/algorithms/greedy_solver.hpp"
#include "optioryx/algorithms/seed_selection_solver.hpp"
#include "optioryx/algorithms/simulated_annealing_solver.hpp"
#include "optioryx/core/evaluator.hpp"
#include "optioryx/io/instance_loader.hpp"
#include "optioryx/io/submission_io.hpp"
#include "optioryx/util/format.hpp"
#include "optioryx/util/output.hpp"

namespace {

using optioryx::Batch;
using optioryx::BatchEvaluator;
using optioryx::BatchStats;
using optioryx::ConvergencePoint;
using optioryx::DecodeResult;
using optioryx::GeneticConfig;
using optioryx::GreedyConfig;
using optioryx::ProblemInstance;
using optioryx::SeedSelectionConfig;
using optioryx::SeedRule;
using optioryx::SelectionRule;
using optioryx::SimulatedAnnealingConfig;
using optioryx::Solution;
using optioryx::StartSolutionStrategy;
using optioryx::SubmissionRow;
using optioryx::WaveSolution;

std::filesystem::path fixture_path(const std::string& name) {
    return std::filesystem::path("fixtures") / name;
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to read test output: " + path.string());
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void expect_contains(const std::string& text, const std::string& expected, const std::string& message) {
    expect(text.find(expected) != std::string::npos, message + " missing token=" + expected);
}

void test_lossless_floating_output() {
    const double precise_objective = std::nextafter(1000000.0, std::numeric_limits<double>::infinity());
    const double precise_runtime = std::nextafter(0.1, 1.0);
    const double precise_control = std::nextafter(0.2, 1.0);
    const double precise_metric = 0.0499583333333333;
    for (const double value : {precise_objective, precise_runtime, precise_control, precise_metric}) {
        const std::string serialized = optioryx::format_double(value);
        expect(std::stod(serialized) == value, "max_digits10 double text should round-trip exactly");
    }

    std::filesystem::create_directories("build");
    const std::filesystem::path json_path = std::filesystem::path("build") / "test_precision_solution.json";
    ProblemInstance instance;
    instance.name = "precision_test";
    Solution solution;
    solution.solver_name = "precision";
    solution.config_id = "precision_config";
    solution.objective = precise_objective;
    solution.runtime_ms = precise_runtime;
    WaveSolution wave;
    wave.wave_id = "wave_precision";
    wave.objective = precise_metric;
    wave.runtime_ms = precise_control;
    wave.convergence.push_back({1, precise_objective, precise_metric, precise_control});
    solution.wave_solutions.push_back(wave);
    optioryx::write_solution_json(json_path, instance, solution);
    const std::string json = read_text_file(json_path);
    expect_contains(json, "\"objective\": " + optioryx::format_double(precise_objective), "JSON objective precision");
    expect_contains(json, "\"runtime_ms\": " + optioryx::format_double(precise_runtime), "JSON runtime precision");
    expect_contains(json, "\"control_value\": " + optioryx::format_double(precise_control), "JSON control precision");

    const std::filesystem::path runs_path = std::filesystem::path("build") / "test_precision_runs.csv";
    optioryx::write_benchmark_runs_csv(
        runs_path,
        {{"precision", "precision_config", "wave_precision", 7, 1, precise_objective, precise_runtime, precise_control, precise_metric}});
    const std::string runs_csv = read_text_file(runs_path);
    expect_contains(runs_csv, optioryx::format_double(precise_objective), "runs.csv objective precision");
    expect_contains(runs_csv, optioryx::format_double(precise_runtime), "runs.csv runtime precision");
    expect_contains(runs_csv, optioryx::format_double(precise_control), "runs.csv metric precision");

    const std::filesystem::path convergence_path =
        std::filesystem::path("build") / "test_precision_convergence.csv";
    optioryx::write_convergence_csv(
        convergence_path,
        {{"precision", "precision_config", "wave_precision", 7, 1, precise_objective, precise_metric, precise_control}});
    const std::string convergence_csv = read_text_file(convergence_path);
    expect_contains(
        convergence_csv,
        optioryx::format_double(precise_control),
        "convergence.csv control precision");

    const ProblemInstance fixture = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(fixture);
    SimulatedAnnealingConfig sa_config;
    sa_config.iterations = 0;
    sa_config.initial_temperature = precise_objective;
    sa_config.final_temperature = precise_metric;
    sa_config.geometric_alpha = precise_runtime;
    sa_config.lundy_beta = precise_control;
    std::mt19937_64 sa_rng(1);
    const auto sa = optioryx::solve_wave_simulated_annealing(fixture, evaluator, 0, sa_config, sa_rng);
    expect(sa.metadata.at("initial_temperature") == optioryx::format_double(precise_objective), "SA initial-temperature metadata precision");
    expect(sa.metadata.at("final_temperature") == optioryx::format_double(precise_metric), "SA final-temperature metadata precision");
    expect(sa.metadata.at("geometric_alpha") == optioryx::format_double(precise_runtime), "SA alpha metadata precision");
    expect(sa.metadata.at("lundy_beta") == optioryx::format_double(precise_control), "SA beta metadata precision");

    GeneticConfig ga_config;
    ga_config.population_size = 6;
    ga_config.generations = 0;
    ga_config.mutation_rate = precise_control;
    ga_config.insertion_rate = precise_runtime;
    ga_config.elitism_fraction = precise_metric;
    std::mt19937_64 ga_rng(2);
    const auto ga = optioryx::solve_wave_genetic(fixture, evaluator, 0, ga_config, ga_rng);
    expect(ga.metadata.at("mutation_rate") == optioryx::format_double(precise_control), "GA mutation metadata precision");
    expect(ga.metadata.at("insertion_rate") == optioryx::format_double(precise_runtime), "GA insertion metadata precision");
    expect(ga.metadata.at("elitism_fraction") == optioryx::format_double(precise_metric), "GA elitism metadata precision");
}

Solution make_solution(
    const ProblemInstance& instance,
    const optioryx::WaveSolveResult& first_wave,
    const optioryx::WaveSolveResult& second_wave) {
    Solution solution;
    solution.solver_name = "test";
    solution.config_id = "test_config";
    solution.wave_solutions.push_back(WaveSolution{
        instance.waves[0].id,
        first_wave.batches,
        first_wave.objective,
        first_wave.runtime_ms,
        first_wave.convergence,
        first_wave.metadata,
    });
    solution.wave_solutions.push_back(WaveSolution{
        instance.waves[1].id,
        second_wave.batches,
        second_wave.objective,
        second_wave.runtime_ms,
        second_wave.convergence,
        second_wave.metadata,
    });
    solution.objective = first_wave.objective + second_wave.objective;
    return solution;
}

void test_batch_costs() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    const BatchStats single = evaluator.batch_stats(0, {0});
    const BatchStats merged = evaluator.batch_stats(0, {0, 1});

    expect_near(single.cost, 6.0, 1e-9, "Standalone order cost mismatch");
    expect_near(merged.cost, 8.0, 1e-9, "Merged batch cost mismatch");
    expect(merged.span_width == 2, "Merged batch span width mismatch");

    const BatchStats canonical = evaluator.batch_stats_sorted_unique(0, {0, 1});
    const BatchStats reordered_with_duplicate = evaluator.batch_stats(0, {1, 0, 1});
    expect_near(canonical.cost, reordered_with_duplicate.cost, 0.0, "Canonical evaluator cost mismatch");
    expect(
        canonical.min_snake_rank == reordered_with_duplicate.min_snake_rank &&
            canonical.max_snake_rank == reordered_with_duplicate.max_snake_rank &&
            canonical.span_width == reordered_with_duplicate.span_width &&
            canonical.unique_item_count == reordered_with_duplicate.unique_item_count,
        "Canonical evaluator statistics mismatch");
}

void test_bounded_batch_evaluator_cache() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator uncached(instance, 0U);
    const BatchEvaluator bounded(instance, 2U);

    struct BatchQuery {
        int wave_index;
        std::vector<int> order_indices;
    };
    const std::vector<BatchQuery> queries = {
        {0, {0}},
        {0, {0, 1}},
        {1, {4}},
        {1, {4, 5}},
        {0, {2}},
        {1, {6}},
    };

    for (const BatchQuery& query : queries) {
        const BatchStats expected = uncached.batch_stats(query.wave_index, query.order_indices);
        const BatchStats actual = bounded.batch_stats(query.wave_index, query.order_indices);
        expect_near(actual.cost, expected.cost, 1e-9, "Cache eviction changed a batch score");
        expect(actual.min_snake_rank == expected.min_snake_rank, "Cache eviction changed the minimum snake rank");
        expect(actual.max_snake_rank == expected.max_snake_rank, "Cache eviction changed the maximum snake rank");
        expect(actual.span_width == expected.span_width, "Cache eviction changed the batch span");
        expect(actual.unique_item_count == expected.unique_item_count, "Cache eviction changed the item count");
        expect(bounded.cache_size() <= 2U, "Global evaluator cache exceeded its configured limit");
        expect(uncached.cache_size() == 0U, "A zero cache limit should disable caching");
    }
}

void test_greedy_and_validation() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    const auto wave_one = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});
    const auto wave_two = optioryx::solve_wave_greedy(instance, evaluator, 1, GreedyConfig{});
    const Solution solution = make_solution(instance, wave_one, wave_two);

    expect_near(wave_one.objective, 22.0, 1e-9, "Wave 1 greedy objective mismatch");
    expect_near(wave_two.objective, 22.0, 1e-9, "Wave 2 greedy objective mismatch");
    expect(evaluator.validate_solution(solution).empty(), "Greedy solution should be feasible");
}

void test_seed_selection() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    SeedSelectionConfig config;
    config.seed_rule = optioryx::SeedRule::largest_snake_span;
    config.selection_rule = SelectionRule::hybrid_score;
    const auto result = optioryx::solve_wave_seed_selection(instance, evaluator, 0, config);

    expect(result.batches.size() == 2U, "Seed-selection should create two batches on wave 1");
    expect_near(result.objective, 22.0, 1e-9, "Seed-selection objective mismatch");
}

void test_seed_selection_paper_adaptations() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    SeedSelectionConfig smallest_config;
    smallest_config.seed_rule = SeedRule::smallest_snake_span;
    smallest_config.selection_rule = SelectionRule::minimum_incremental_cost;
    const auto smallest = optioryx::solve_wave_seed_selection(instance, evaluator, 0, smallest_config);
    expect(
        smallest.batches.front().order_indices.front() == 2,
        "Smallest snake-span should seed with the first zero-span order");

    SeedSelectionConfig overlap_config;
    overlap_config.seed_rule = SeedRule::largest_snake_span;
    overlap_config.selection_rule = SelectionRule::maximum_location_overlap;
    const auto overlap = optioryx::solve_wave_seed_selection(instance, evaluator, 0, overlap_config);
    expect(overlap.batches.front().order_indices.front() == 0, "Largest snake-span seed mismatch");
    expect(
        overlap.batches.front().order_indices.at(1) == 1,
        "Maximum location overlap should add the order sharing location B");

    expect(
        optioryx::parse_seed_rule("span") == SeedRule::largest_snake_span,
        "Legacy seed-rule alias should remain supported");
    expect(
        optioryx::parse_selection_rule("hybrid") == SelectionRule::hybrid_score,
        "Legacy selection-rule alias should remain supported");
    expect(
        optioryx::parse_seed_rule("smallest-snake-span") == SeedRule::smallest_snake_span,
        "Smallest snake-span parser mismatch");
    expect(
        optioryx::parse_selection_rule("maximum-location-overlap") == SelectionRule::maximum_location_overlap,
        "Maximum overlap parser mismatch");
    expect(
        optioryx::to_string(SelectionRule::bounded_two_step_lookahead) == "bounded_two_step_lookahead",
        "Lookahead formatter mismatch");
}

ProblemInstance make_lookahead_fixture() {
    ProblemInstance instance;
    instance.name = "lookahead_fixture";
    instance.nmax = 3U;
    instance.start_index = 0;
    instance.end_index = 0;
    instance.item_ids = {"DEPOT", "S", "A", "B"};
    instance.snake_rank = {0, 1, 2, 3};

    const std::vector<std::pair<double, double>> coordinates = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {10.0, 0.0},
    };
    instance.distances.assign(coordinates.size(), std::vector<double>(coordinates.size(), 0.0));
    for (std::size_t from = 0; from < coordinates.size(); ++from) {
        for (std::size_t to = 0; to < coordinates.size(); ++to) {
            const double dx = coordinates[from].first - coordinates[to].first;
            const double dy = coordinates[from].second - coordinates[to].second;
            instance.distances[from][to] = std::sqrt(dx * dx + dy * dy);
        }
    }

    instance.orders = {
        {"seed", "wave", {1}},
        {"near", "wave", {2}},
        {"far_a", "wave", {3}},
        {"far_b", "wave", {3}},
    };
    instance.waves = {{"wave", {0, 1, 2, 3}}};
    return instance;
}

void test_bounded_two_step_lookahead() {
    const ProblemInstance instance = make_lookahead_fixture();
    const BatchEvaluator evaluator(instance);

    SeedSelectionConfig immediate_config;
    immediate_config.seed_rule = SeedRule::smallest_snake_span;
    immediate_config.selection_rule = SelectionRule::minimum_incremental_cost;
    const auto immediate = optioryx::solve_wave_seed_selection(instance, evaluator, 0, immediate_config);
    expect(
        immediate.batches.front().order_indices.at(1) == 1,
        "Immediate-cost ablation should select the nearby order first");

    SeedSelectionConfig width_one_config;
    width_one_config.seed_rule = SeedRule::smallest_snake_span;
    width_one_config.selection_rule = SelectionRule::bounded_two_step_lookahead;
    width_one_config.lookahead_width = 1;
    const auto width_one = optioryx::solve_wave_seed_selection(instance, evaluator, 0, width_one_config);
    expect(
        width_one.batches.front().order_indices.at(1) == 1,
        "Lookahead width one should retain the best immediate candidate");

    SeedSelectionConfig width_two_config = width_one_config;
    width_two_config.lookahead_width = 2;
    const auto width_two = optioryx::solve_wave_seed_selection(instance, evaluator, 0, width_two_config);
    expect(
        width_two.batches.front().order_indices.at(1) == 2,
        "Two-step lookahead should prefer the candidate with the better paired batch cost");
    expect(
        width_two.metadata.at("lookahead_width") == "2",
        "Lookahead width should be recorded in solver metadata");

    SeedSelectionConfig invalid_config = width_one_config;
    invalid_config.lookahead_width = 0;
    bool rejected_invalid_width = false;
    try {
        static_cast<void>(optioryx::solve_wave_seed_selection(instance, evaluator, 0, invalid_config));
    } catch (const std::invalid_argument&) {
        rejected_invalid_width = true;
    }
    expect(rejected_invalid_width, "Non-positive lookahead width should be rejected");
}

void test_simulated_annealing_regression() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);
    const auto greedy = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});

    SimulatedAnnealingConfig config;
    config.start_strategy = StartSolutionStrategy::greedy;
    config.iterations = 80;
    std::mt19937_64 rng(7);
    const auto result = optioryx::solve_wave_simulated_annealing(instance, evaluator, 0, config, rng);

    expect(result.convergence.size() == 81U, "SA should emit one convergence point per iteration plus start");
    expect(result.objective <= greedy.objective + 1e-9, "SA should not be worse than greedy with greedy start");
    expect(
        result.metadata.at("greedy_start_local_search_passes") == "1" &&
            result.metadata.at("greedy_start_local_search_neighbor_window") == "8",
        "SA should record its explicit strong greedy-start configuration");
}

void test_simulated_annealing_explicit_pure_greedy_start() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    GreedyConfig pure_greedy_config;
    pure_greedy_config.local_search_passes = 0;
    pure_greedy_config.local_search_neighbor_window = 0;
    const auto pure_greedy = optioryx::solve_wave_greedy(instance, evaluator, 0, pure_greedy_config);

    SimulatedAnnealingConfig config;
    config.start_strategy = StartSolutionStrategy::greedy;
    config.greedy_start_local_search_passes = 0;
    config.greedy_start_local_search_neighbor_window = 0;
    config.iterations = 0;
    std::mt19937_64 rng(13);
    const auto result = optioryx::solve_wave_simulated_annealing(instance, evaluator, 0, config, rng);

    expect_near(
        result.objective,
        pure_greedy.objective,
        1e-9,
        "SA LS0/w0 start should exactly match the pure greedy comparator before search");
    expect(
        result.metadata.at("greedy_start_local_search_passes") == "0" &&
            result.metadata.at("greedy_start_local_search_neighbor_window") == "0",
        "SA should record its explicit pure greedy-start configuration");
}

void test_simulated_annealing_randomized_start_is_capacity_matched() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);

    SimulatedAnnealingConfig config;
    config.start_strategy = StartSolutionStrategy::randomized;
    config.iterations = 0;

    for (std::size_t wave_index = 0U; wave_index < instance.waves.size(); ++wave_index) {
        const std::vector<int>& expected_orders = instance.waves[wave_index].order_indices;
        const std::size_t expected_batch_count =
            (expected_orders.size() + instance.nmax - 1U) / instance.nmax;
        std::vector<int> first_layout;
        bool observed_distinct_layout = false;

        // Elke seed moet uitsluitend de ordervolgorde randomiseren. Routeaantal,
        // capaciteit en volledigheid van de toewijzing blijven invariant.
        for (const std::uint64_t seed : {0ULL, 1ULL, 2ULL, 7ULL, 42ULL, 999ULL}) {
            std::mt19937_64 rng(seed);
            const auto result = optioryx::solve_wave_simulated_annealing(
                instance,
                evaluator,
                static_cast<int>(wave_index),
                config,
                rng);

            expect(
                result.batches.size() == expected_batch_count,
                "Randomized SA start should use the capacity-minimal batch count");

            std::vector<int> actual_orders;
            for (std::size_t batch_index = 0U; batch_index < result.batches.size(); ++batch_index) {
                const Batch& batch = result.batches[batch_index];
                expect(!batch.order_indices.empty(), "Randomized SA start should not create empty batches");
                expect(
                    batch.order_indices.size() <= instance.nmax,
                    "Randomized SA start should respect nmax");
                if (batch_index + 1U < result.batches.size()) {
                    expect(
                        batch.order_indices.size() == instance.nmax,
                        "Randomized SA start should fill every non-final batch to capacity");
                }
                actual_orders.insert(
                    actual_orders.end(),
                    batch.order_indices.begin(),
                    batch.order_indices.end());
            }

            if (first_layout.empty()) {
                first_layout = actual_orders;
            } else if (actual_orders != first_layout) {
                observed_distinct_layout = true;
            }

            std::vector<int> sorted_expected = expected_orders;
            std::sort(actual_orders.begin(), actual_orders.end());
            std::sort(sorted_expected.begin(), sorted_expected.end());
            expect(
                actual_orders == sorted_expected,
                "Randomized SA start should assign each wave order exactly once");
        }
        expect(
            observed_distinct_layout,
            "Different SA seeds should be able to produce different randomized starts");
    }

    std::mt19937_64 first_rng(123456);
    std::mt19937_64 repeated_rng(123456);
    const auto first = optioryx::solve_wave_simulated_annealing(instance, evaluator, 0, config, first_rng);
    const auto repeated = optioryx::solve_wave_simulated_annealing(instance, evaluator, 0, config, repeated_rng);
    expect(first.batches.size() == repeated.batches.size(), "Equal SA seeds should reproduce the batch count");
    for (std::size_t batch_index = 0U; batch_index < first.batches.size(); ++batch_index) {
        expect(
            first.batches[batch_index].order_indices == repeated.batches[batch_index].order_indices,
            "Equal SA seeds should reproduce the randomized start exactly");
    }
}

void test_simulated_annealing_incremental_objective() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const std::vector<optioryx::NeighborhoodStrategy> neighborhoods = {
        optioryx::NeighborhoodStrategy::relocate,
        optioryx::NeighborhoodStrategy::swap,
        optioryx::NeighborhoodStrategy::merge_split,
        optioryx::NeighborhoodStrategy::mixed,
    };
    const std::vector<StartSolutionStrategy> starts = {
        StartSolutionStrategy::greedy,
        StartSolutionStrategy::randomized,
    };

    for (const optioryx::NeighborhoodStrategy neighborhood : neighborhoods) {
        for (const StartSolutionStrategy start : starts) {
            // Een evaluator zonder cache dwingt elke gewijzigde batch werkelijk
            // opnieuw te berekenen en verbergt dus geen fout in de delta-administratie.
            const BatchEvaluator evaluator(instance, 0U);
            SimulatedAnnealingConfig config;
            config.start_strategy = start;
            config.neighborhood = neighborhood;
            config.iterations = 500;
            config.initial_temperature = 100.0;
            config.geometric_alpha = 0.99;

            std::mt19937_64 rng(12345);
            const auto result = optioryx::solve_wave_simulated_annealing(instance, evaluator, 0, config, rng);
            const double independently_scored = evaluator.wave_cost(0, result.batches);

            expect_near(
                result.objective,
                independently_scored,
                1e-8,
                "Incremental SA objective diverged from an independent full evaluation");
            expect(
                result.convergence.size() == static_cast<std::size_t>(config.iterations + 1),
                "Incremental SA should preserve the convergence output cardinality");
            expect_near(
                result.convergence.back().best_objective,
                result.objective,
                1e-8,
                "Final SA convergence value should equal the returned best objective");

            std::size_t assigned_orders = 0U;
            for (const Batch& batch : result.batches) {
                expect(!batch.order_indices.empty(), "SA should never retain an empty batch");
                expect(batch.order_indices.size() <= instance.nmax, "SA delta move violated batch capacity");
                assigned_orders += batch.order_indices.size();
            }
            expect(
                assigned_orders == instance.waves[0].order_indices.size(),
                "SA delta move lost or duplicated an order");
        }
    }
}

void test_ga_decoder_and_regression() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);
    const auto decoded = optioryx::decode_permutation_to_batches(instance, evaluator, 0, {0, 1, 2, 3});
    const auto greedy = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});

    expect(decoded.batches.size() == 2U, "GA decoder should split wave 1 into two batches");
    expect_near(decoded.objective, 22.0, 1e-9, "GA decoder objective mismatch");

    GeneticConfig config;
    expect(config.greedy_start_local_search_passes == 1, "GA should default to the historical LS1 greedy start");
    expect(
        config.greedy_start_local_search_neighbor_window == 8,
        "GA should default to the historical width-8 greedy start");
    config.population_size = 16;
    config.generations = 40;
    std::mt19937_64 rng(11);
    const auto result = optioryx::solve_wave_genetic(instance, evaluator, 0, config, rng);

    expect(!result.convergence.empty(), "GA should emit convergence data");
    expect(result.objective <= greedy.objective + 1e-9, "GA should not be worse than greedy on the small fixture");
    expect(
        result.metadata.at("greedy_start_local_search_passes") == "1",
        "GA metadata should expose its greedy-start local-search passes");
    expect(
        result.metadata.at("greedy_start_local_search_neighbor_window") == "8",
        "GA metadata should expose its greedy-start neighbour window");

    GeneticConfig pure_start_config;
    pure_start_config.population_size = 2;
    pure_start_config.generations = 0;
    pure_start_config.greedy_start_local_search_passes = 0;
    pure_start_config.greedy_start_local_search_neighbor_window = 0;
    GreedyConfig pure_greedy_config;
    pure_greedy_config.local_search_passes = 0;
    pure_greedy_config.local_search_neighbor_window = 0;
    const auto pure_greedy = optioryx::solve_wave_greedy(instance, evaluator, 0, pure_greedy_config);
    std::mt19937_64 pure_start_rng(17);
    const auto pure_start_result =
        optioryx::solve_wave_genetic(instance, evaluator, 0, pure_start_config, pure_start_rng);
    expect(
        pure_start_result.objective <= pure_greedy.objective + 1e-9,
        "Generation-zero GA should contain the configured greedy warm start");
    expect(
        pure_start_result.metadata.at("greedy_start_local_search_passes") == "0" &&
            pure_start_result.metadata.at("greedy_start_local_search_neighbor_window") == "0",
        "GA should preserve an explicitly disabled local-search warm start in metadata");
}

void test_submission_roundtrip() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);
    const auto wave_one = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});
    const auto wave_two = optioryx::solve_wave_greedy(instance, evaluator, 1, GreedyConfig{});
    Solution solution = make_solution(instance, wave_one, wave_two);
    solution.feasible = true;

    std::filesystem::create_directories("build");
    const std::filesystem::path submission_path = std::filesystem::path("build") / "test_submission.csv";
    optioryx::write_submission_csv(submission_path, instance, solution);
    const std::vector<SubmissionRow> rows = optioryx::load_submission_csv(submission_path);
    const std::vector<std::string> errors = optioryx::validate_submission_rows(instance, rows);

    expect(errors.empty(), "Generated submission should validate");
    expect_near(optioryx::score_submission_rows(instance, rows), solution.objective, 1e-9, "Submission score mismatch");
}

void test_submission_validation_rejects_common_errors() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);
    const auto wave_one = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});
    const auto wave_two = optioryx::solve_wave_greedy(instance, evaluator, 1, GreedyConfig{});
    Solution solution = make_solution(instance, wave_one, wave_two);
    solution.feasible = true;

    const std::vector<SubmissionRow> valid_rows = optioryx::build_submission_rows(instance, solution);

    std::vector<SubmissionRow> duplicate_rows = valid_rows;
    duplicate_rows.push_back(valid_rows.front());
    expect(!optioryx::validate_submission_rows(instance, duplicate_rows).empty(), "Duplicate ORDER_ID should be rejected");

    std::vector<SubmissionRow> missing_rows = valid_rows;
    missing_rows.pop_back();
    expect(!optioryx::validate_submission_rows(instance, missing_rows).empty(), "Missing ORDER_ID should be rejected");

    std::vector<SubmissionRow> mixed_wave_rows = valid_rows;
    const std::string first_list = mixed_wave_rows.front().list_id;
    for (SubmissionRow& row : mixed_wave_rows) {
        if (row.wave_id != mixed_wave_rows.front().wave_id) {
            row.list_id = first_list;
            break;
        }
    }
    expect(!optioryx::validate_submission_rows(instance, mixed_wave_rows).empty(), "LIST_ID across waves should be rejected");

    std::vector<SubmissionRow> over_capacity_rows = valid_rows;
    int changed = 0;
    for (SubmissionRow& row : over_capacity_rows) {
        if (row.wave_id == "wave_1" && changed < 3) {
            row.list_id = "too-large";
            ++changed;
        }
    }
    expect(!optioryx::validate_submission_rows(instance, over_capacity_rows).empty(), "LIST_ID above nmax should be rejected");
}

void test_competition_loader_if_available() {
    if (!std::filesystem::exists(std::filesystem::path("data") / "pick_data.csv") ||
        !std::filesystem::exists(std::filesystem::path("data") / "dist_matrix.csv")) {
        return;
    }

    const ProblemInstance instance = optioryx::load_instance("data");
    expect(instance.orders.size() == 10246U, "Competition loader should read all orders");
    expect(instance.waves.size() == 12U, "Competition loader should read all waves");
    expect(instance.item_ids.size() == 422U, "Competition loader should read all locations");
    expect(instance.start_index != instance.end_index, "Competition loader should keep distinct START and END");

    // Stress de delta-administratie ook op een echte, grote wave. De cache staat
    // uit zodat de onafhankelijke eindscore volledig opnieuw wordt opgebouwd.
    const int wave_index = static_cast<int>(instance.waves.size() - 1U);
    const BatchEvaluator evaluator(instance, 0U);
    SimulatedAnnealingConfig config;
    config.start_strategy = StartSolutionStrategy::randomized;
    config.neighborhood = optioryx::NeighborhoodStrategy::mixed;
    config.iterations = 2000;
    config.initial_temperature = 100.0;
    config.geometric_alpha = 0.995;
    std::mt19937_64 rng(8675309);
    const auto result = optioryx::solve_wave_simulated_annealing(instance, evaluator, wave_index, config, rng);
    expect_near(
        result.objective,
        evaluator.wave_cost(wave_index, result.batches),
        1e-6,
        "Incremental SA objective diverged on the competition-scale wave");
}

}  // namespace

int main() {
    try {
        test_lossless_floating_output();
        test_batch_costs();
        test_bounded_batch_evaluator_cache();
        test_greedy_and_validation();
        test_seed_selection();
        test_seed_selection_paper_adaptations();
        test_bounded_two_step_lookahead();
        test_simulated_annealing_regression();
        test_simulated_annealing_explicit_pure_greedy_start();
        test_simulated_annealing_randomized_start_is_capacity_matched();
        test_simulated_annealing_incremental_objective();
        test_ga_decoder_and_regression();
        test_submission_roundtrip();
        test_submission_validation_rejects_common_errors();
        test_competition_loader_if_available();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << "\n";
        return 1;
    }
}
