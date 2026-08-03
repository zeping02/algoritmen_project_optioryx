#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "optioryx/util/format.hpp"
#include "optioryx/util/output.hpp"

namespace optioryx {
namespace {

// Een benchmarkconfiguratie: solvernaam, key-value config en de seeds waarmee
// stochastic solvers herhaald worden.
struct BenchmarkSpec {
    std::string solver;
    ConfigMap config;
    std::vector<std::uint64_t> seeds;
};

// Een record op instantieniveau voorkomt dat variantie tussen waves wordt
// verward met variantie tussen onafhankelijke runs van een stochastische solver.
struct InstanceBenchmarkRecord {
    std::string solver;
    std::string config_id;
    std::uint64_t seed = 0;
    int order_count = 0;
    double objective = 0.0;
    double runtime_ms = 0.0;
    bool feasible = false;
};

// Kleine command-line helpers. De CLI gebruikt expliciete opties zoals
// --input en --solver zodat experimenten reproduceerbaar in scripts kunnen staan.
std::optional<std::string> find_option(const std::vector<std::string>& args, const std::string& name) {
    for (std::size_t index = 0; index + 1U < args.size(); ++index) {
        if (args[index] == name) {
            return args[index + 1U];
        }
    }
    return std::nullopt;
}

bool has_flag(const std::vector<std::string>& args, const std::string& name) {
    return std::find(args.begin(), args.end(), name) != args.end();
}

std::string require_option(const std::vector<std::string>& args, const std::string& name) {
    const std::optional<std::string> value = find_option(args, name);
    if (!value.has_value()) {
        throw std::runtime_error("Missing required option: " + name);
    }
    return *value;
}

bool matches_solver(const std::string& requested, const std::string& canonical) {
    // Enkele korte aliassen blijven ondersteund voor sneller experimenteren.
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
    // Los elke wave apart op. Dat is correct omdat de opdracht verbiedt om orders
    // uit verschillende waves in dezelfde picklijst te mengen.
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
            // Alle greedy-parameters komen uit het configbestand met veilige defaults.
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
            greedy_config.triple_rebuild_passes = config_int(config, "triple_rebuild_passes", 0);
            greedy_config.triple_rebuild_neighbor_window =
                config_int(config, "triple_rebuild_neighbor_window", greedy_config.pair_rebuild_neighbor_window);
            greedy_config.triple_rebuild_combination_window =
                config_int(config, "triple_rebuild_combination_window", 0);
            greedy_config.post_triple_pair_rebuild_passes = config_int(config, "post_triple_pair_rebuild_passes", 0);
            greedy_config.post_triple_pair_rebuild_neighbor_window = config_int(
                config,
                "post_triple_pair_rebuild_neighbor_window",
                greedy_config.pair_rebuild_neighbor_window);
            greedy_config.post_pair_triple_rebuild_passes = config_int(config, "post_pair_triple_rebuild_passes", 0);
            greedy_config.post_pair_triple_rebuild_neighbor_window = config_int(
                config,
                "post_pair_triple_rebuild_neighbor_window",
                greedy_config.triple_rebuild_neighbor_window);
            greedy_config.post_pair_triple_rebuild_combination_window = config_int(
                config,
                "post_pair_triple_rebuild_combination_window",
                greedy_config.triple_rebuild_combination_window);
            greedy_config.post_second_triple_pair_rebuild_passes =
                config_int(config, "post_second_triple_pair_rebuild_passes", 0);
            greedy_config.post_second_triple_pair_rebuild_neighbor_window = config_int(
                config,
                "post_second_triple_pair_rebuild_neighbor_window",
                greedy_config.post_triple_pair_rebuild_neighbor_window);
            greedy_config.final_local_search_passes = config_int(config, "final_local_search_passes", 0);
            solution.config_id = greedy_config.config_id;
            result = solve_wave_greedy(instance, evaluator, static_cast<int>(wave_index), greedy_config);
        } else if (matches_solver(solver_name, "simulated_annealing")) {
            // SA en GA gebruiken dezelfde evaluator, zodat scores eerlijk vergelijkbaar zijn.
            SimulatedAnnealingConfig sa_config;
            sa_config.config_id = config_string(config, "config_id", "sa_default");
            sa_config.start_strategy = parse_start_strategy(config_string(config, "start_strategy", "greedy"));
            sa_config.greedy_start_local_search_passes =
                config_int(config, "greedy_start_local_search_passes", 1);
            sa_config.greedy_start_local_search_neighbor_window =
                config_int(config, "greedy_start_local_search_neighbor_window", 8);
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
            ga_config.greedy_start_local_search_passes =
                config_int(config, "greedy_start_local_search_passes", 1);
            ga_config.greedy_start_local_search_neighbor_window =
                config_int(config, "greedy_start_local_search_neighbor_window", 8);
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
            seed_config.lookahead_width = config_int(config, "lookahead_width", 8);
            solution.config_id = seed_config.config_id;
            result = solve_wave_seed_selection(instance, evaluator, static_cast<int>(wave_index), seed_config);
        } else {
            throw std::runtime_error("Unknown solver: " + solver_name);
        }

        solution.wave_solutions.push_back(to_wave_solution(wave.id, result));
        solution.objective += result.objective;
    }

    // Validatie gebeurt na het samenvoegen van alle wave-oplossingen.
    solution.validation_errors = evaluator.validate_solution(solution);
    solution.feasible = solution.validation_errors.empty();
    const auto finished = std::chrono::steady_clock::now();
    solution.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return solution;
}

std::vector<std::uint64_t> make_seeds(int count) {
    std::vector<std::uint64_t> seeds;
    seeds.reserve(static_cast<std::size_t>(count));
    for (int seed = 0; seed < count; ++seed) {
        seeds.push_back(static_cast<std::uint64_t>(seed));
    }
    return seeds;
}

std::string id_value(std::string value) {
    // Punten in numerieke waarden worden bestandsvriendelijk gemaakt. De exacte,
    // parseerbare waarde blijft daarnaast in suite_manifest.csv staan.
    std::replace(value.begin(), value.end(), '.', 'p');
    return value;
}

ConfigMap make_greedy_baseline_config() {
    // Ook parameters die nul zijn staan expliciet in de config. Zo kan een
    // vergelijking nooit ongemerkt veranderen wanneer een solver-default wijzigt.
    return {
        {"config_id", "greedy_ls-1_win-8_elim-0_pair-0_triple-0_final-0"},
        {"local_search_passes", "1"},
        {"local_search_neighbor_window", "8"},
        {"route_elimination_passes", "0"},
        {"route_elimination_max_batch_size", "12"},
        {"route_elimination_beam_width", "1"},
        {"route_elimination_candidates_per_order", "1"},
        {"route_elimination_ordering_trials", "1"},
        {"post_elimination_local_search_passes", "0"},
        {"pair_rebuild_passes", "0"},
        {"pair_rebuild_neighbor_window", "8"},
        {"triple_rebuild_passes", "0"},
        {"triple_rebuild_neighbor_window", "8"},
        {"triple_rebuild_combination_window", "0"},
        {"post_triple_pair_rebuild_passes", "0"},
        {"post_triple_pair_rebuild_neighbor_window", "8"},
        {"post_pair_triple_rebuild_passes", "0"},
        {"post_pair_triple_rebuild_neighbor_window", "8"},
        {"post_pair_triple_rebuild_combination_window", "0"},
        {"post_second_triple_pair_rebuild_passes", "0"},
        {"post_second_triple_pair_rebuild_neighbor_window", "8"},
        {"final_local_search_passes", "0"},
    };
}

ConfigMap make_greedy_fast_config() {
    // Praktische savings-baseline zonder lokale zoekpass. Alle afgeleide
    // neighbourhood-windows staan expliciet op nul in plaats van defaults te erven.
    ConfigMap config = make_greedy_baseline_config();
    config["config_id"] = "greedy_fast_ls-0_win-0_elim-0_pair-0_triple-0_final-0";
    config["local_search_passes"] = "0";
    config["local_search_neighbor_window"] = "0";
    config["pair_rebuild_neighbor_window"] = "0";
    config["triple_rebuild_neighbor_window"] = "0";
    config["post_triple_pair_rebuild_neighbor_window"] = "0";
    config["post_pair_triple_rebuild_neighbor_window"] = "0";
    config["post_second_triple_pair_rebuild_neighbor_window"] = "0";
    return config;
}

ConfigMap make_sa_config(
    const std::string& cooling,
    const std::string& neighborhood,
    const std::string& start_strategy,
    int greedy_start_local_search_passes,
    int greedy_start_local_search_neighbor_window,
    const std::string& iterations,
    const std::string& geometric_alpha,
    const std::string& lundy_beta) {
    std::string start_label = start_strategy;
    if (start_strategy == "greedy") {
        start_label +=
            "-ls-" + std::to_string(greedy_start_local_search_passes) +
            "-win-" + std::to_string(greedy_start_local_search_neighbor_window);
    }
    const std::string config_id =
        "sa_c-" + cooling +
        "_n-" + neighborhood +
        "_start-" + start_label +
        "_i-" + iterations +
        "_t0-60_tf-0p05_a-" + id_value(geometric_alpha) +
        "_b-" + id_value(lundy_beta);
    return {
        {"config_id", config_id},
        {"cooling", cooling},
        {"neighborhood", neighborhood},
        {"start_strategy", start_strategy},
        {"greedy_start_local_search_passes", std::to_string(greedy_start_local_search_passes)},
        {"greedy_start_local_search_neighbor_window", std::to_string(greedy_start_local_search_neighbor_window)},
        {"iterations", iterations},
        {"initial_temperature", "60"},
        {"final_temperature", "0.05"},
        {"geometric_alpha", geometric_alpha},
        {"lundy_beta", lundy_beta},
    };
}

ConfigMap make_assignment_sa_config(
    const std::string& cooling,
    const std::string& neighborhood,
    const std::string& start_strategy) {
    // Beide schema's zijn gekalibreerd van T0=60 naar Tf=0.05 na 400
    // updates. Daardoor vergelijken we schema-vorm, niet een andere eindtemperatuur.
    return make_sa_config(
        cooling,
        neighborhood,
        start_strategy,
        1,
        8,
        "400",
        "0.9824309750689336",
        "0.0499583333333333");
}

ConfigMap make_ga_config(
    int population_size,
    int generations,
    const std::string& mutation_rate,
    const std::string& elitism_fraction,
    int greedy_start_local_search_passes = 1,
    int greedy_start_local_search_neighbor_window = 8) {
    const std::string population = std::to_string(population_size);
    const std::string generation_count = std::to_string(generations);
    const std::string config_id =
        "ga_start-greedy-ls-" + std::to_string(greedy_start_local_search_passes) +
        "-win-" + std::to_string(greedy_start_local_search_neighbor_window) +
        "_pop-" + population +
        "_gen-" + generation_count +
        "_tour-3_mut-" + id_value(mutation_rate) +
        "_insert-0p50_elite-" + id_value(elitism_fraction);
    return {
        {"config_id", config_id},
        {"greedy_start_local_search_passes", std::to_string(greedy_start_local_search_passes)},
        {"greedy_start_local_search_neighbor_window", std::to_string(greedy_start_local_search_neighbor_window)},
        {"population_size", population},
        {"generations", generation_count},
        {"tournament_size", "3"},
        {"mutation_rate", mutation_rate},
        {"insertion_rate", "0.50"},
        {"elitism_fraction", elitism_fraction},
    };
}

ConfigMap make_seed_config(
    const std::string& seed_rule,
    const std::string& selection_rule,
    int lookahead_width = 8) {
    std::string selection_label = selection_rule;
    if (selection_rule == "hybrid_score") {
        // Alleen voor backwards-compatible oude suites/configs.
        selection_label = "hybrid_score_cost-0p70_span-0p30";
    }
    if (selection_rule == "bounded_two_step_lookahead") {
        selection_label += "_width-" + std::to_string(lookahead_width);
    }
    ConfigMap config = {
        {"config_id", "seed_seed-" + seed_rule + "_select-" + selection_label},
        {"seed_rule", seed_rule},
        {"selection_rule", selection_rule},
    };
    if (selection_rule == "bounded_two_step_lookahead") {
        config["lookahead_width"] = std::to_string(lookahead_width);
    }
    return config;
}

std::vector<BenchmarkSpec> build_sa_factorial_suite() {
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    std::vector<BenchmarkSpec> specs;
    for (const std::string& cooling : {std::string("geometric"), std::string("lundy_mees")}) {
        for (const std::string& neighborhood : {std::string("relocate"), std::string("swap")}) {
            for (const std::string& start : {std::string("greedy"), std::string("randomized")}) {
                specs.push_back({"simulated_annealing", make_assignment_sa_config(cooling, neighborhood, start), seeds});
            }
        }
    }
    return specs;
}

std::vector<BenchmarkSpec> build_ga_sensitivity_suite() {
    // One-factor-at-a-time rond dezelfde expliciete baseline. Populatie en
    // generaties veranderen bewust het werkbudget; runtime/kwaliteit wordt mee gemeten.
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    std::vector<BenchmarkSpec> specs;
    specs.push_back({"genetic", make_ga_config(40, 120, "0.20", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 60, "0.20", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 240, "0.20", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(20, 120, "0.20", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(80, 120, "0.20", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 120, "0.05", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 120, "0.40", "0.05"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 120, "0.20", "0.01"), seeds});
    specs.push_back({"genetic", make_ga_config(40, 120, "0.20", "0.15"), seeds});
    return specs;
}

std::vector<BenchmarkSpec> build_ga_sensitivity_fast_suite() {
    // Zelfde OFAT-opzet als de academische studie, maar rond de praktische
    // competition-fast baseline zodat de volledige data haalbaar blijft.
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    std::vector<BenchmarkSpec> specs;
    specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 10, "0.20", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 40, "0.20", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(6, 20, "0.20", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(24, 20, "0.20", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 20, "0.05", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 20, "0.40", "0.08"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0"), seeds});
    specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0.20"), seeds});
    return specs;
}

std::vector<BenchmarkSpec> build_seed_factorial_suite() {
    std::vector<BenchmarkSpec> specs;
    const std::vector<std::string> seed_rules = {"smallest_snake_span", "largest_snake_span"};
    const std::vector<std::string> paper_selection_rules = {
        "maximum_location_overlap",
        "minimum_span_expansion",
    };
    for (const std::string& seed_rule : seed_rules) {
        for (const std::string& selection_rule : paper_selection_rules) {
            specs.push_back({"seed_selection", make_seed_config(seed_rule, selection_rule), {0}});
        }
    }
    // De eigen regel en haar directe-kostenablation worden onder exact dezelfde
    // twee seedregels uitgevoerd; K=8 staat expliciet in manifest en config-id.
    for (const std::string& seed_rule : seed_rules) {
        specs.push_back({"seed_selection", make_seed_config(seed_rule, "bounded_two_step_lookahead", 8), {0}});
    }
    for (const std::string& seed_rule : seed_rules) {
        specs.push_back({"seed_selection", make_seed_config(seed_rule, "minimum_incremental_cost"), {0}});
    }
    return specs;
}

std::vector<BenchmarkSpec> build_comparison_suite() {
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    return {
        {"greedy", make_greedy_baseline_config(), {0}},
        {"simulated_annealing", make_assignment_sa_config("geometric", "relocate", "greedy"), seeds},
        {"genetic", make_ga_config(40, 120, "0.20", "0.05"), seeds},
        {"seed_selection", make_seed_config("largest_snake_span", "bounded_two_step_lookahead", 8), {0}},
    };
}

std::vector<BenchmarkSpec> build_comparison_fast_suite() {
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    return {
        {"greedy", make_greedy_fast_config(), {0}},
        {"simulated_annealing",
         make_sa_config(
             "geometric",
             "relocate",
             "greedy",
             0,
             0,
             "120",
             "0.9426276099699734",
             "0.1665277777777778"),
         seeds},
        {"genetic", make_ga_config(12, 20, "0.20", "0.08"), seeds},
        {"seed_selection", make_seed_config("largest_snake_span", "bounded_two_step_lookahead", 8), {0}},
    };
}

std::vector<BenchmarkSpec> build_comparison_fast_matched_suite() {
    const std::vector<std::uint64_t> seeds = make_seeds(10);
    return {
        {"greedy", make_greedy_baseline_config(), {0}},
        {"simulated_annealing",
         make_sa_config(
             "geometric",
             "relocate",
             "greedy",
             1,
             8,
             "120",
             "0.9426276099699734",
             "0.1665277777777778"),
         seeds},
        {"genetic", make_ga_config(12, 20, "0.20", "0.08", 1, 8), seeds},
        {"seed_selection", make_seed_config("largest_snake_span", "bounded_two_step_lookahead", 8), {0}},
    };
}

void append_unique_specs(std::vector<BenchmarkSpec>& destination, const std::vector<BenchmarkSpec>& additions) {
    for (const BenchmarkSpec& addition : additions) {
        const std::string& config_id = addition.config.at("config_id");
        const auto existing = std::find_if(destination.begin(), destination.end(), [&](const BenchmarkSpec& candidate) {
            return candidate.solver == addition.solver && candidate.config.at("config_id") == config_id;
        });
        if (existing == destination.end()) {
            destination.push_back(addition);
            continue;
        }
        if (existing->config != addition.config || existing->seeds != addition.seeds) {
            throw std::runtime_error("Conflicting benchmark definitions for config_id: " + config_id);
        }
    }
}

std::string canonical_suite_name(std::string suite_name) {
    std::replace(suite_name.begin(), suite_name.end(), '-', '_');
    if (suite_name == "all") {
        return "assignment";
    }
    return suite_name;
}

std::vector<BenchmarkSpec> build_benchmark_suite(const std::string& requested_suite_name) {
    const std::string suite_name = canonical_suite_name(requested_suite_name);
    if (suite_name == "smoke") {
        return {
            {"greedy", make_greedy_baseline_config(), {0}},
            {"simulated_annealing",
             make_sa_config(
                 "geometric",
                 "relocate",
                 "greedy",
                 1,
                 8,
                 "120",
                 "0.9426276099699734",
                 "0.1665277777777778"),
             {0, 1}},
            {"genetic", make_ga_config(20, 40, "0.20", "0.05"), {0, 1}},
            {"seed_selection", make_seed_config("highest_standalone_cost", "minimum_incremental_cost"), {0}},
        };
    }
    if (suite_name == "sa_factorial") {
        return build_sa_factorial_suite();
    }
    if (suite_name == "ga_sensitivity") {
        return build_ga_sensitivity_suite();
    }
    if (suite_name == "ga_sensitivity_fast") {
        return build_ga_sensitivity_fast_suite();
    }
    if (suite_name == "seed_factorial") {
        return build_seed_factorial_suite();
    }
    if (suite_name == "comparison") {
        return build_comparison_suite();
    }
    if (suite_name == "comparison_fast") {
        return build_comparison_fast_suite();
    }
    if (suite_name == "comparison_fast_matched") {
        return build_comparison_fast_matched_suite();
    }
    if (suite_name == "assignment") {
        std::vector<BenchmarkSpec> specs;
        append_unique_specs(specs, build_sa_factorial_suite());
        append_unique_specs(specs, build_ga_sensitivity_suite());
        append_unique_specs(specs, build_seed_factorial_suite());
        append_unique_specs(specs, build_comparison_suite());
        return specs;
    }

    throw std::runtime_error(
        "Unknown benchmark suite: " + requested_suite_name +
        ". Expected smoke, sa_factorial, ga_sensitivity, ga_sensitivity_fast, seed_factorial, "
        "comparison, comparison_fast, comparison_fast_matched, or assignment.");
}

std::string csv_field(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2U);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string join_seeds(const std::vector<std::uint64_t>& seeds) {
    std::ostringstream output;
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        if (index > 0U) {
            output << ';';
        }
        output << seeds[index];
    }
    return output.str();
}

std::string serialize_config(const ConfigMap& config) {
    std::vector<std::pair<std::string, std::string>> parameters(config.begin(), config.end());
    std::sort(parameters.begin(), parameters.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    std::ostringstream output;
    bool first = true;
    for (const auto& [key, value] : parameters) {
        if (!first) {
            output << ';';
        }
        first = false;
        output << key << '=' << value;
    }
    return output.str();
}

std::string convergence_control_metric(const std::string& solver) {
    if (solver == "simulated_annealing") {
        return "temperature";
    }
    if (solver == "genetic") {
        return "population_diversity_mean_normalized_hamming";
    }
    return "none";
}

int expected_convergence_points(const BenchmarkSpec& spec) {
    if (spec.solver == "simulated_annealing") {
        return config_int(spec.config, "iterations", 400) + 1;
    }
    if (spec.solver == "genetic") {
        return config_int(spec.config, "generations", 120) + 1;
    }
    return 0;
}

void write_suite_manifest(
    const std::filesystem::path& path,
    const std::string& suite_name,
    const std::vector<BenchmarkSpec>& specs,
    std::size_t wave_count) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write suite manifest: " + path.string());
    }
    output << "suite,solver,config_id,seed_count,instance_runs,wave_runs,seeds,parameters,"
              "convergence_control,convergence_points_per_wave\n";
    for (const BenchmarkSpec& spec : specs) {
        const std::size_t instance_runs = spec.seeds.size();
        output << csv_field(suite_name) << ','
               << csv_field(spec.solver) << ','
               << csv_field(spec.config.at("config_id")) << ','
               << spec.seeds.size() << ','
               << instance_runs << ','
               << instance_runs * wave_count << ','
               << csv_field(join_seeds(spec.seeds)) << ','
               << csv_field(serialize_config(spec.config)) << ','
               << csv_field(convergence_control_metric(spec.solver)) << ','
               << expected_convergence_points(spec) << '\n';
    }
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double sample_standard_deviation(const std::vector<double>& values) {
    if (values.size() < 2U) {
        // Een sample-SD uit een enkele observatie is niet nul maar onbepaald.
        // De CSV schrijft dit als nan, wat pandas als ontbrekende waarde leest.
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double average = mean(values);
    double squared_deviations = 0.0;
    for (const double value : values) {
        const double difference = value - average;
        squared_deviations += difference * difference;
    }
    return std::sqrt(squared_deviations / static_cast<double>(values.size() - 1U));
}

void write_instance_runs_csv(
    const std::filesystem::path& path,
    const std::vector<InstanceBenchmarkRecord>& records) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write instance benchmark CSV: " + path.string());
    }
    configure_lossless_floating_output(output);
    output << "solver,config_id,seed,order_count,objective,runtime_ms,feasible\n";
    for (const InstanceBenchmarkRecord& record : records) {
        output << record.solver << ','
               << record.config_id << ','
               << record.seed << ','
               << record.order_count << ','
               << record.objective << ','
               << record.runtime_ms << ','
               << (record.feasible ? "true" : "false") << '\n';
    }
}

void write_summary(
    const std::filesystem::path& out_dir,
    const std::vector<InstanceBenchmarkRecord>& records) {
    // Statistiek over onafhankelijke volledige-instantie-runs. Dit rapporteert
    // spreiding over seeds zonder de verschillende waves als replicaten te behandelen.
    std::ofstream output(out_dir / "summary.txt");
    if (!output) {
        throw std::runtime_error("Unable to write summary.txt");
    }
    configure_lossless_floating_output(output);

    using Key = std::pair<std::string, std::string>;
    std::map<Key, std::vector<const InstanceBenchmarkRecord*>> grouped;
    for (const InstanceBenchmarkRecord& record : records) {
        grouped[{record.solver, record.config_id}].push_back(&record);
    }

    output << "solver,config_id,runs,mean_objective,std_objective,best_objective,worst_objective,"
              "mean_runtime_ms,std_runtime_ms,feasible_runs\n";
    for (const auto& [key, group] : grouped) {
        std::vector<double> objectives;
        std::vector<double> runtimes;
        int feasible_runs = 0;
        for (const InstanceBenchmarkRecord* record : group) {
            objectives.push_back(record->objective);
            runtimes.push_back(record->runtime_ms);
            feasible_runs += record->feasible ? 1 : 0;
        }
        const auto [minimum, maximum] = std::minmax_element(objectives.begin(), objectives.end());
        output << key.first << ','
               << key.second << ','
               << group.size() << ','
               << mean(objectives) << ','
               << sample_standard_deviation(objectives) << ','
               << *minimum << ','
               << *maximum << ','
               << mean(runtimes) << ','
               << sample_standard_deviation(runtimes) << ','
               << feasible_runs << '\n';
    }
}

void write_wave_summary(
    const std::filesystem::path& path,
    const std::vector<BenchmarkRecord>& records) {
    using Key = std::tuple<std::string, std::string, std::string>;
    std::map<Key, std::vector<const BenchmarkRecord*>> grouped;
    for (const BenchmarkRecord& record : records) {
        grouped[{record.solver, record.config_id, record.wave_id}].push_back(&record);
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write wave summary CSV: " + path.string());
    }
    configure_lossless_floating_output(output);
    output << "solver,config_id,wave_id,order_count,runs,mean_objective,std_objective,best_objective,"
              "worst_objective,mean_runtime_ms,std_runtime_ms,overlap_density,span_dispersion\n";
    for (const auto& [key, group] : grouped) {
        std::vector<double> objectives;
        std::vector<double> runtimes;
        for (const BenchmarkRecord* record : group) {
            objectives.push_back(record->objective);
            runtimes.push_back(record->runtime_ms);
        }
        const auto [minimum, maximum] = std::minmax_element(objectives.begin(), objectives.end());
        const BenchmarkRecord& first = *group.front();
        output << std::get<0>(key) << ','
               << std::get<1>(key) << ','
               << std::get<2>(key) << ','
               << first.order_count << ','
               << group.size() << ','
               << mean(objectives) << ','
               << sample_standard_deviation(objectives) << ','
               << *minimum << ','
               << *maximum << ','
               << mean(runtimes) << ','
               << sample_standard_deviation(runtimes) << ','
               << first.overlap_density << ','
               << first.span_dispersion << '\n';
    }
}

void write_ga_diversity_csv(
    const std::filesystem::path& path,
    const std::vector<ConvergenceRecord>& records) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write GA diversity CSV: " + path.string());
    }
    configure_lossless_floating_output(output);
    output << "solver,config_id,wave_id,seed,generation,best_objective,generation_best_objective,"
              "population_diversity_mean_normalized_hamming\n";
    for (const ConvergenceRecord& record : records) {
        if (record.solver != "genetic") {
            continue;
        }
        output << record.solver << ','
               << record.config_id << ','
               << record.wave_id << ','
               << record.seed << ','
               << record.step << ','
               << record.best_objective << ','
               << record.current_objective << ','
               << record.control_value << '\n';
    }
}

bool is_submission_output(const std::filesystem::path& path) {
    return path.extension() == ".csv";
}

int run_solve_mode(const std::vector<std::string>& args) {
    // Produceert ofwel een Kaggle-compatible CSV, ofwel een JSON met extra metadata.
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
    // Schrijft de convergentiepunten van iteratieve algoritmes naar CSV voor grafieken.
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
    // Controleert een bestaande submission onafhankelijk van de solver die ze maakte.
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
    // Draait een suite van solvers/configs en schrijft zowel ruwe runs als
    // convergentiedata weg voor analyse en visualisatie.
    const std::filesystem::path input_path = require_option(args, "--input");
    const std::filesystem::path out_dir = require_option(args, "--out-dir");
    const std::string suite_name = canonical_suite_name(find_option(args, "--suite").value_or("assignment"));
    const ProblemInstance instance = load_instance(input_path);
    const std::vector<BenchmarkSpec> specs = build_benchmark_suite(suite_name);

    std::filesystem::create_directories(out_dir);
    write_suite_manifest(out_dir / "suite_manifest.csv", suite_name, specs, instance.waves.size());

    std::size_t planned_instance_runs = 0U;
    for (const BenchmarkSpec& spec : specs) {
        planned_instance_runs += spec.seeds.size();
    }
    std::cout << "Benchmark suite: " << suite_name << "\n";
    std::cout << "Configurations: " << specs.size() << "\n";
    std::cout << "Planned instance runs: " << planned_instance_runs << "\n";
    std::cout << "Planned wave runs: " << planned_instance_runs * instance.waves.size() << "\n";
    std::cout << "Manifest: " << (out_dir / "suite_manifest.csv").string() << "\n";
    if (has_flag(args, "--dry-run")) {
        std::cout << "Dry run complete; no solver was executed.\n";
        return 0;
    }

    std::filesystem::create_directories(out_dir / "solutions");
    BatchEvaluator evaluator(instance);

    std::vector<BenchmarkRecord> benchmark_records;
    std::vector<InstanceBenchmarkRecord> instance_records;
    std::vector<ConvergenceRecord> convergence_records;

    std::size_t completed_instance_runs = 0U;
    for (const BenchmarkSpec& spec : specs) {
        for (const std::uint64_t seed : spec.seeds) {
            const Solution solution = solve_instance(instance, spec.solver, spec.config, seed);
            ++completed_instance_runs;
            std::cout << '[' << completed_instance_runs << '/' << planned_instance_runs << "] "
                      << solution.solver_name << ' ' << solution.config_id << " seed=" << seed << "\n";
            const std::filesystem::path solution_path =
                out_dir / "solutions" / (solution.solver_name + "_" + solution.config_id + "_seed" + std::to_string(seed) + ".json");
            write_solution_json(solution_path, instance, solution);

            instance_records.push_back(
                {solution.solver_name,
                 solution.config_id,
                 seed,
                 static_cast<int>(instance.orders.size()),
                 solution.objective,
                 solution.runtime_ms,
                 solution.feasible});

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
    write_instance_runs_csv(out_dir / "instance_runs.csv", instance_records);
    write_convergence_csv(out_dir / "convergence.csv", convergence_records);
    write_ga_diversity_csv(out_dir / "ga_diversity.csv", convergence_records);
    write_wave_summary(out_dir / "wave_summary.csv", benchmark_records);
    write_summary(out_dir, instance_records);
    std::cout << "Benchmark suite complete: " << suite_name << "\n";
    std::cout << "Runs written to: " << (out_dir / "runs.csv").string() << "\n";
    return 0;
}

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  optioryx_cli solve --solver <name> --input <path> --output <path> [--config <path>] [--seed <n>]\n";
    std::cout << "  optioryx_cli benchmark --input <path> --out-dir <path> [--suite <name>] [--dry-run]\n";
    std::cout << "    suites: smoke, sa_factorial, ga_sensitivity, ga_sensitivity_fast, seed_factorial,\n";
    std::cout << "            comparison, comparison_fast, comparison_fast_matched, assignment\n";
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
