#include "optioryx/util/output.hpp"

#include "optioryx/util/format.hpp"

#include <fstream>
#include <stdexcept>

namespace optioryx {

namespace {

// Minimale JSON-escaping voor strings die in analyse-output terechtkomen.
std::string escape_json(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

void ensure_parent_directory(const std::filesystem::path& path) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

}  // namespace

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write file: " + path.string());
    }
    output << content;
}

void write_solution_json(const std::filesystem::path& path, const ProblemInstance& instance, const Solution& solution) {
    // JSON-output is bedoeld voor debugging en analyse, niet voor Kaggle.
    // Daarom bevat hij ook metadata, batches en convergentiepunten.
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write solution JSON: " + path.string());
    }
    configure_lossless_floating_output(output);

    output << "{\n";
    output << "  \"instance\": \"" << escape_json(instance.name) << "\",\n";
    output << "  \"solver\": \"" << escape_json(solution.solver_name) << "\",\n";
    output << "  \"config_id\": \"" << escape_json(solution.config_id) << "\",\n";
    output << "  \"seed\": " << solution.seed << ",\n";
    output << "  \"objective\": " << solution.objective << ",\n";
    output << "  \"runtime_ms\": " << solution.runtime_ms << ",\n";
    output << "  \"feasible\": " << (solution.feasible ? "true" : "false") << ",\n";
    output << "  \"validation_errors\": [";
    for (std::size_t index = 0; index < solution.validation_errors.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << "\"" << escape_json(solution.validation_errors[index]) << "\"";
    }
    output << "],\n";
    output << "  \"waves\": [\n";

    for (std::size_t wave_index = 0; wave_index < solution.wave_solutions.size(); ++wave_index) {
        const WaveSolution& wave_solution = solution.wave_solutions[wave_index];
        output << "    {\n";
        output << "      \"wave_id\": \"" << escape_json(wave_solution.wave_id) << "\",\n";
        output << "      \"objective\": " << wave_solution.objective << ",\n";
        output << "      \"runtime_ms\": " << wave_solution.runtime_ms << ",\n";
        output << "      \"metadata\": {";
        bool first_metadata = true;
        for (const auto& [key, value] : wave_solution.metadata) {
            if (!first_metadata) {
                output << ", ";
            }
            first_metadata = false;
            output << "\"" << escape_json(key) << "\": \"" << escape_json(value) << "\"";
        }
        output << "},\n";
        output << "      \"batches\": [\n";

        for (std::size_t batch_index = 0; batch_index < wave_solution.batches.size(); ++batch_index) {
            const Batch& batch = wave_solution.batches[batch_index];
            output << "        {\"wave_id\": \"" << escape_json(batch.wave_id) << "\", \"orders\": [";
            for (std::size_t order_pos = 0; order_pos < batch.order_indices.size(); ++order_pos) {
                if (order_pos > 0) {
                    output << ", ";
                }
                const Order& order = instance.orders.at(static_cast<std::size_t>(batch.order_indices[order_pos]));
                output << "\"" << escape_json(order.id) << "\"";
            }
            output << "]}";
            if (batch_index + 1U != wave_solution.batches.size()) {
                output << ",";
            }
            output << "\n";
        }
        output << "      ],\n";
        output << "      \"convergence\": [\n";
        for (std::size_t point_index = 0; point_index < wave_solution.convergence.size(); ++point_index) {
            const ConvergencePoint& point = wave_solution.convergence[point_index];
            output << "        {\"step\": " << point.step
                   << ", \"best_objective\": " << point.best_objective
                   << ", \"current_objective\": " << point.current_objective
                   << ", \"control_value\": " << point.control_value << "}";
            if (point_index + 1U != wave_solution.convergence.size()) {
                output << ",";
            }
            output << "\n";
        }
        output << "      ]\n";
        output << "    }";
        if (wave_index + 1U != solution.wave_solutions.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]\n";
    output << "}\n";
}

void write_benchmark_runs_csv(const std::filesystem::path& path, const std::vector<BenchmarkRecord>& records) {
    // Platte CSV-output maakt het gemakkelijk om resultaten later in Python,
    // spreadsheets of LaTeX-tabellen te gebruiken.
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write benchmark CSV: " + path.string());
    }
    configure_lossless_floating_output(output);

    output << "solver,config_id,wave_id,seed,order_count,objective,runtime_ms,overlap_density,span_dispersion\n";
    for (const BenchmarkRecord& record : records) {
        output << record.solver << ','
               << record.config_id << ','
               << record.wave_id << ','
               << record.seed << ','
               << record.order_count << ','
               << record.objective << ','
               << record.runtime_ms << ','
               << record.overlap_density << ','
               << record.span_dispersion << '\n';
    }
}

void write_convergence_csv(const std::filesystem::path& path, const std::vector<ConvergenceRecord>& records) {
    // Convergentiegegevens ondersteunen grafieken voor SA/GA en benchmarkanalyse.
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write convergence CSV: " + path.string());
    }
    configure_lossless_floating_output(output);

    output << "solver,config_id,wave_id,seed,step,best_objective,current_objective,control_value\n";
    for (const ConvergenceRecord& record : records) {
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

}  // namespace optioryx
