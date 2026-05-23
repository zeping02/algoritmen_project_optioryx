#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "optioryx/algorithms/genetic_solver.hpp"
#include "optioryx/algorithms/greedy_solver.hpp"
#include "optioryx/algorithms/seed_selection_solver.hpp"
#include "optioryx/algorithms/simulated_annealing_solver.hpp"
#include "optioryx/core/analysis.hpp"
#include "optioryx/core/evaluator.hpp"
#include "optioryx/io/config_loader.hpp"
#include "optioryx/io/instance_loader.hpp"
#include "optioryx/io/submission_io.hpp"
#include "optioryx/util/output.hpp"

namespace optioryx {
namespace {

struct BenchmarkSpec {
    std::string solver;
    ConfigMap config;
    std::vector<std::uint64_t> seeds;
};

std::optional<std::string> find_option(const std::vector<std::string>& args, const std::string& name) {
    for (std::size_t index = 0; index + 1U < args.size(); ++index) {
        if (args[index] == name) {
            return args[index + 1U];
        }
    }
    return std::nullopt;
}

std::string require_option(const std::vector<std::string>& args, const std::string& name) {
    const std::optional<std::string> value = find_option(args, name);
    if (!value.has_value()) {
        throw std::runtime_error("Missing required option: " + name);
    }
    return *value;
}

bool matches_solver(const std::string& requested, const std::string& canonical) {
    if (requested == canonical) {
        return true;
    }
    if (canonical == "simulated_annealing" && requested == "sa") {
        return true;
    }
    if (canonical == "seed_selection" && (requested == "seed" || requested == "paper")) {
        return true;
    }
    return false;
}

WaveSolution to_wave_solution(const std::string& wave_id, const WaveSolveResult& result) {
    WaveSolution wave_solution;
    wave_solution.wave_id = wave_id;
    wave_solution.batches = result.batches;
    wave_solution.objective = result.objective;
    wave_solution.runtime_ms = result.runtime_ms;
    wave_solution.convergence = result.convergence;
    wave_solution.metadata = result.metadata;
    return wave_solution;
}

Solution solve_instance(
    const ProblemInstance& instance,
    const std::string& solver_name,
    const ConfigMap& config,
    std::uint64_t seed) {
    const auto started = std::chrono::steady_clock::now();
    BatchEvaluator evaluator(instance);
    Solution solution;
    solution.solver_name = solver_name;
    solution.seed = seed;

    std::mt19937_64 rng(seed);
    for (std::size_t wave_index = 0; wave_index < instance.waves.size(); ++wave_index) {
        const Wave& wave = instance.waves[wave_index];
        WaveSolveResult result;

        if (matches_solver(solver_name, "greedy")) {
            GreedyConfig greedy_config;
            greedy_config.config_id = config_string(config, "config_id", "greedy_default");
            greedy_config.local_search_passes = config_int(config, "local_search_passes", 1);
            greedy_config.local_search_neighbor_window = config_int(config, "local_search_neighbor_window", 8);
            greedy_config.route_elimination_passes = config_int(config, "route_elimination_passes", 0);
            greedy_config.route_elimination_max_batch_size = config_int(config, "route_elimination_max_batch_size", 12);
            greedy_config.route_elimination_beam_width = config_int(config, "route_elimination_beam_width", 1);
            greedy_config.route_elimination_candidates_per_order =
                config_int(config, "route_elimination_candidates_per_order", 1);
            greedy_config.route_elimination_ordering_trials = config_int(config, "route_elimination_ordering_trials", 1);
            greedy_config.post_elimination_local_search_passes = config_int(config, "post_elimination_local_search_passes", 0);
            greedy_config.pair_rebuild_passes = config_int(config, "pair_rebuild_passes", 0);
            greedy_config.pair_rebuild_neighbor_window =
                config_int(config, "pair_rebuild_neighbor_window", greedy_config.local_search_neighbor_window);
            greedy_config.final_local_search_passes = config_int(config, "final_local_search_passes", 0);
            solution.config_id = greedy_config.config_id;
            result = solve_wave_greedy(instance, evaluator, static_cast<int>(wave_index), greedy_config);
        } else if (matches_solver(solver_name, "simulated_annealing")) {
            SimulatedAnnealingConfig sa_config;
            sa_config.config_id = config_string(config, "config_id", "sa_default");
            sa_config.start_strategy = parse_start_strategy(config_string(config, "start_strategy", "greedy"));
            sa_config.neighborhood = parse_neighborhood_strategy(config_string(config, "neighborhood", "relocate"));
            sa_config.cooling = parse_cooling_schedule(config_string(config, "cooling", "geometric"));
            sa_config.iterations = config_int(config, "iterations", 400);
            sa_config.initial_temperature = config_double(config, "initial_temperature", 60.0);
            sa_config.final_temperature = config_double(config, "final_temperature", 0.05);
            sa_config.geometric_alpha = config_double(config, "geometric_alpha", 0.97);
            sa_config.lundy_beta = config_double(config, "lundy_beta", 0.01);
            solution.config_id = sa_config.config_id;
            result = solve_wave_simulated_annealing(instance, evaluator, static_cast<int>(wave_index), sa_config, rng);
        } else if (matches_solver(solver_name, "genetic")) {
            GeneticConfig ga_config;
            ga_config.config_id = config_string(config, "config_id", "ga_default");
            ga_config.population_size = config_int(config, "population_size", 40);
            ga_config.generations = config_int(config, "generations", 120);
            ga_config.tournament_size = config_int(config, "tournament_size", 3);
            ga_config.mutation_rate = config_double(config, "mutation_rate", 0.2);
            ga_config.insertion_rate = config_double(config, "insertion_rate", 0.5);
            ga_config.elitism_fraction = config_double(config, "elitism_fraction", 0.05);
            solution.config_id = ga_config.config_id;
            result = solve_wave_genetic(instance, evaluator, static_cast<int>(wave_index), ga_config, rng);
        } else if (matches_solver(solver_name, "seed_selection")) {
            SeedSelectionConfig seed_config;
            seed_config.config_id = config_string(config, "config_id", "seed_selection_default");
            seed_config.seed_rule = parse_seed_rule(config_string(config, "seed_rule", "cost"));
            seed_config.selection_rule = parse_selection_rule(config_string(config, "selection_rule", "incremental_cost"));
            solution.config_id = seed_config.config_id;
            result = solve_wave_seed_selection(instance, evaluator, static_cast<int>(wave_index), seed_config);
        } else {
            throw std::runtime_error("Unknown solver: " + solver_name);
        }

        solution.wave_solutions.push_back(to_wave_solution(wave.id, result));
        solution.objective += result.objective;
    }

    solution.validation_errors = evaluator.validate_solution(solution);
    solution.feasible = solution.validation_errors.empty();
    const auto finished = std::chrono::steady_clock::now();
    solution.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return solution;
}

std::vector<BenchmarkSpec> build_benchmark_suite(const std::string& suite_name) {
    auto make_seeds = [](int count) {
        std::vector<std::uint64_t> seeds;
        seeds.reserve(static_cast<std::size_t>(count));
        for (int seed = 0; seed < count; ++seed) {
            seeds.push_back(static_cast<std::uint64_t>(seed));
        }
        return seeds;
    };

    if (suite_name == "smoke") {
        return {
            {"greedy", {{"config_id", "greedy_default"}}, {0}},
            {"simulated_annealing",
             {{"config_id", "sa_geo_relocate_greedy"}, {"cooling", "geometric"}, {"neighborhood", "relocate"}, {"start_strategy", "greedy"}, {"iterations", "120"}},
             {0, 1}},
            {"genetic", {{"config_id", "ga_smoke"}, {"population_size", "20"}, {"generations", "40"}}, {0, 1}},
            {"seed_selection", {{"config_id", "seed_cost_incremental"}, {"seed_rule", "cost"}, {"selection_rule", "incremental_cost"}}, {0}},
        };
    }

    if (suite_name == "assignment") {
        const std::vector<std::uint64_t> stochastic_seeds = make_seeds(10);
        return {
            {"greedy", {{"config_id", "greedy_default"}}, {0}},
            {"simulated_annealing",
             {{"config_id", "sa_geo_relocate_greedy"}, {"cooling", "geometric"}, {"neighborhood", "relocate"}, {"start_strategy", "greedy"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_geo_relocate_randomized"}, {"cooling", "geometric"}, {"neighborhood", "relocate"}, {"start_strategy", "randomized"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_geo_swap_greedy"}, {"cooling", "geometric"}, {"neighborhood", "swap"}, {"start_strategy", "greedy"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_geo_swap_randomized"}, {"cooling", "geometric"}, {"neighborhood", "swap"}, {"start_strategy", "randomized"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_lundy_relocate_greedy"}, {"cooling", "lundy_mees"}, {"neighborhood", "relocate"}, {"start_strategy", "greedy"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_lundy_relocate_randomized"}, {"cooling", "lundy_mees"}, {"neighborhood", "relocate"}, {"start_strategy", "randomized"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_lundy_swap_greedy"}, {"cooling", "lundy_mees"}, {"neighborhood", "swap"}, {"start_strategy", "greedy"}},
             stochastic_seeds},
            {"simulated_annealing",
             {{"config_id", "sa_lundy_swap_randomized"}, {"cooling", "lundy_mees"}, {"neighborhood", "swap"}, {"start_strategy", "randomized"}},
             stochastic_seeds},
            {"genetic", {{"config_id", "ga_default"}}, stochastic_seeds},
            {"seed_selection", {{"config_id", "seed_cost_incremental"}, {"seed_rule", "cost"}, {"selection_rule", "incremental_cost"}}, {0}},
            {"seed_selection", {{"config_id", "seed_cost_span"}, {"seed_rule", "cost"}, {"selection_rule", "span_expansion"}}, {0}},
            {"seed_selection", {{"config_id", "seed_span_incremental"}, {"seed_rule", "span"}, {"selection_rule", "incremental_cost"}}, {0}},
            {"seed_selection", {{"config_id", "seed_span_hybrid"}, {"seed_rule", "span"}, {"selection_rule", "hybrid"}}, {0}},
        };
    }

    throw std::runtime_error("Unknown benchmark suite: " + suite_name);
}

void write_summary(
    const std::filesystem::path& out_dir,
    const std::vector<BenchmarkRecord>& records) {
    std::ofstream output(out_dir / "summary.txt");
    if (!output) {
        throw std::runtime_error("Unable to write summary.txt");
    }

    output << "solver,config_id,mean_objective,mean_runtime_ms\n";
    std::map<std::pair<std::string, std::string>, std::pair<double, double>> sums;
    std::map<std::pair<std::string, std::string>, int> counts;
    for (const BenchmarkRecord& record : records) {
        const auto key = std::make_pair(record.solver, record.config_id);
        sums[key].first += record.objective;
        sums[key].second += record.runtime_ms;
        counts[key] += 1;
    }

    for (const auto& [key, values] : sums) {
        const int count = counts[key];
        output << key.first << ','
               << key.second << ','
               << values.first / static_cast<double>(count) << ','
               << values.second / static_cast<double>(count) << '\n';
    }
}

bool is_submission_output(const std::filesystem::path& path) {
    return path.extension() == ".csv";
}

int run_solve_mode(const std::vector<std::string>& args) {
    const std::filesystem::path input_path = require_option(args, "--input");
    const std::filesystem::path output_path = require_option(args, "--output");
    const std::string solver = require_option(args, "--solver");
    const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(find_option(args, "--seed").value_or("0")));

    ConfigMap config;
    if (const auto config_path = find_option(args, "--config")) {
        config = load_config(*config_path);
    }

    const ProblemInstance instance = load_instance(input_path);
    const Solution solution = solve_instance(instance, solver, config, seed);
    if (is_submission_output(output_path)) {
        write_submission_csv(output_path, instance, solution);
    } else {
        write_solution_json(output_path, instance, solution);
    }

    std::cout << "Solver: " << solution.solver_name << "\n";
    std::cout << "Config: " << solution.config_id << "\n";
    std::cout << "Objective: " << solution.objective << "\n";
    std::cout << "Runtime ms: " << solution.runtime_ms << "\n";
    std::cout << "Feasible: " << (solution.feasible ? "yes" : "no") << "\n";
    if (is_submission_output(output_path)) {
        std::cout << "Submission rows: " << instance.orders.size() << "\n";
    }
    return solution.feasible ? 0 : 2;
}

int run_dump_convergence_mode(const std::vector<std::string>& args) {
    const std::filesystem::path input_path = require_option(args, "--input");
    const std::filesystem::path output_path = require_option(args, "--output");
    const std::string solver = require_option(args, "--solver");
    const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(find_option(args, "--seed").value_or("0")));

    ConfigMap config;
    if (const auto config_path = find_option(args, "--config")) {
        config = load_config(*config_path);
    }

    const ProblemInstance instance = load_instance(input_path);
    const Solution solution = solve_instance(instance, solver, config, seed);

    std::vector<ConvergenceRecord> records;
    for (const WaveSolution& wave_solution : solution.wave_solutions) {
        for (const ConvergencePoint& point : wave_solution.convergence) {
            records.push_back(
                {solution.solver_name, solution.config_id, wave_solution.wave_id, solution.seed, point.step, point.best_objective, point.current_objective, point.control_value});
        }
    }

    write_convergence_csv(output_path, records);
    return 0;
}

int run_validate_submission_mode(const std::vector<std::string>& args) {
    const std::filesystem::path input_path = require_option(args, "--input");
    const std::filesystem::path submission_path = require_option(args, "--submission");

    const ProblemInstance instance = load_instance(input_path);
    const std::vector<SubmissionRow> rows = load_submission_csv(submission_path);
    const std::vector<std::string> errors = validate_submission_rows(instance, rows);

    std::cout << "Submission rows: " << rows.size() << "\n";
    if (!errors.empty()) {
        std::cout << "Valid submission: no\n";
        for (const std::string& error : errors) {
            std::cout << " - " << error << "\n";
        }
        return 2;
    }

    std::cout << "Valid submission: yes\n";
    std::cout << "Local score: " << score_submission_rows(instance, rows) << "\n";
    return 0;
}

int run_benchmark_mode(const std::vector<std::string>& args) {
    const std::filesystem::path input_path = require_option(args, "--input");
    const std::filesystem::path out_dir = require_option(args, "--out-dir");
    const std::string suite_name = find_option(args, "--suite").value_or("assignment");
    const ProblemInstance instance = load_instance(input_path);
    BatchEvaluator evaluator(instance);
    const std::vector<BenchmarkSpec> specs = build_benchmark_suite(suite_name);

    std::filesystem::create_directories(out_dir / "solutions");

    std::vector<BenchmarkRecord> benchmark_records;
    std::vector<ConvergenceRecord> convergence_records;

    for (const BenchmarkSpec& spec : specs) {
        for (const std::uint64_t seed : spec.seeds) {
            const Solution solution = solve_instance(instance, spec.solver, spec.config, seed);
            const std::filesystem::path solution_path =
                out_dir / "solutions" / (solution.solver_name + "_" + solution.config_id + "_seed" + std::to_string(seed) + ".json");
            write_solution_json(solution_path, instance, solution);

            for (std::size_t wave_index = 0; wave_index < solution.wave_solutions.size(); ++wave_index) {
                const WaveSolution& wave_solution = solution.wave_solutions[wave_index];
                const WaveMetrics metrics = compute_wave_metrics(instance, evaluator, static_cast<int>(wave_index));
                benchmark_records.push_back(
                    {solution.solver_name,
                     solution.config_id,
                     wave_solution.wave_id,
                     seed,
                     static_cast<int>(instance.waves[wave_index].order_indices.size()),
                     wave_solution.objective,
                     wave_solution.runtime_ms,
                     metrics.overlap_density,
                     metrics.span_dispersion});

                for (const ConvergencePoint& point : wave_solution.convergence) {
                    convergence_records.push_back(
                        {solution.solver_name,
                         solution.config_id,
                         wave_solution.wave_id,
                         seed,
                         point.step,
                         point.best_objective,
                         point.current_objective,
                         point.control_value});
                }
            }
        }
    }

    write_benchmark_runs_csv(out_dir / "runs.csv", benchmark_records);
    write_convergence_csv(out_dir / "convergence.csv", convergence_records);
    write_summary(out_dir, benchmark_records);
    std::cout << "Benchmark suite complete: " << suite_name << "\n";
    std::cout << "Runs written to: " << (out_dir / "runs.csv").string() << "\n";
    return 0;
}

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  optioryx_cli solve --solver <name> --input <path> --output <path> [--config <path>] [--seed <n>]\n";
    std::cout << "  optioryx_cli benchmark --input <path> --out-dir <path> [--suite <name>]\n";
    std::cout << "  optioryx_cli dump-convergence --solver <name> --input <path> --output <path> [--config <path>] [--seed <n>]\n";
    std::cout << "  optioryx_cli validate-submission --input <path> --submission <path>\n";
}

}  // namespace
}  // namespace optioryx

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc));
        for (int index = 1; index < argc; ++index) {
            args.emplace_back(argv[index]);
        }

        if (args.empty()) {
            optioryx::print_usage();
            return 0;
        }

        const std::string mode = args.front();
        if (mode == "solve") {
            return optioryx::run_solve_mode(args);
        }
        if (mode == "benchmark") {
            return optioryx::run_benchmark_mode(args);
        }
        if (mode == "dump-convergence") {
            return optioryx::run_dump_convergence_mode(args);
        }
        if (mode == "validate-submission") {
            return optioryx::run_validate_submission_mode(args);
        }

        optioryx::print_usage();
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }
}
