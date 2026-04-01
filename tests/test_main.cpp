#include <filesystem>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include "optioryx/algorithms/genetic_solver.hpp"
#include "optioryx/algorithms/greedy_solver.hpp"
#include "optioryx/algorithms/seed_selection_solver.hpp"
#include "optioryx/algorithms/simulated_annealing_solver.hpp"
#include "optioryx/core/evaluator.hpp"
#include "optioryx/io/instance_loader.hpp"

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
using optioryx::SelectionRule;
using optioryx::SimulatedAnnealingConfig;
using optioryx::Solution;
using optioryx::StartSolutionStrategy;
using optioryx::WaveSolution;

std::filesystem::path fixture_path(const std::string& name) {
    return std::filesystem::path("data") / "fixtures" / name;
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
}

void test_ga_decoder_and_regression() {
    const ProblemInstance instance = optioryx::load_instance(fixture_path("small.instance"));
    const BatchEvaluator evaluator(instance);
    const auto decoded = optioryx::decode_permutation_to_batches(instance, evaluator, 0, {0, 1, 2, 3});
    const auto greedy = optioryx::solve_wave_greedy(instance, evaluator, 0, GreedyConfig{});

    expect(decoded.batches.size() == 2U, "GA decoder should split wave 1 into two batches");
    expect_near(decoded.objective, 22.0, 1e-9, "GA decoder objective mismatch");

    GeneticConfig config;
    config.population_size = 16;
    config.generations = 40;
    std::mt19937_64 rng(11);
    const auto result = optioryx::solve_wave_genetic(instance, evaluator, 0, config, rng);

    expect(!result.convergence.empty(), "GA should emit convergence data");
    expect(result.objective <= greedy.objective + 1e-9, "GA should not be worse than greedy on the small fixture");
}

}  // namespace

int main() {
    try {
        test_batch_costs();
        test_greedy_and_validation();
        test_seed_selection();
        test_simulated_annealing_regression();
        test_ga_decoder_and_regression();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << "\n";
        return 1;
    }
}
