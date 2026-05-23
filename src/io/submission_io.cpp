#include "optioryx/io/submission_io.hpp"

#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "optioryx/core/evaluator.hpp"
#include "optioryx/io/csv_utils.hpp"

namespace optioryx {

namespace {

std::string make_list_id(const std::string& wave_id, int list_index) {
    std::ostringstream stream;
    stream << wave_id << "-ALG-L" << std::setw(3) << std::setfill('0') << list_index;
    return stream.str();
}

void ensure_parent_directory(const std::filesystem::path& path) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::size_t require_column(const std::vector<std::string>& header, const std::string& column_name) {
    std::size_t match = header.size();
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (header[index] == column_name) {
            if (match != header.size()) {
                throw std::runtime_error("Duplicate column in submission CSV: " + column_name);
            }
            match = index;
        }
    }
    if (match == header.size()) {
        throw std::runtime_error("Submission CSV is missing required column: " + column_name);
    }
    return match;
}

}  // namespace

std::vector<SubmissionRow> build_submission_rows(const ProblemInstance& instance, const Solution& solution) {
    BatchEvaluator evaluator(instance);
    const std::vector<std::string> errors = evaluator.validate_solution(solution);
    if (!errors.empty()) {
        throw std::runtime_error("Cannot build submission from an invalid solution");
    }

    std::vector<std::string> list_id_by_order(instance.orders.size());
    for (const WaveSolution& wave_solution : solution.wave_solutions) {
        for (std::size_t batch_index = 0; batch_index < wave_solution.batches.size(); ++batch_index) {
            const std::string list_id = make_list_id(wave_solution.wave_id, static_cast<int>(batch_index));
            for (const int order_index : wave_solution.batches[batch_index].order_indices) {
                list_id_by_order.at(static_cast<std::size_t>(order_index)) = list_id;
            }
        }
    }

    std::vector<SubmissionRow> rows;
    rows.reserve(instance.orders.size());
    for (std::size_t order_index = 0; order_index < instance.orders.size(); ++order_index) {
        if (list_id_by_order[order_index].empty()) {
            throw std::runtime_error("Order is missing a LIST_ID assignment in the generated submission");
        }
        rows.push_back(
            {instance.orders[order_index].wave_id, instance.orders[order_index].id, list_id_by_order[order_index]});
    }
    return rows;
}

void write_submission_csv(const std::filesystem::path& path, const ProblemInstance& instance, const Solution& solution) {
    ensure_parent_directory(path);
    const std::vector<SubmissionRow> rows = build_submission_rows(instance, solution);

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write submission CSV: " + path.string());
    }

    output << "WAVE_ID,ORDER_ID,LIST_ID\n";
    for (const SubmissionRow& row : rows) {
        output << row.wave_id << ',' << row.order_id << ',' << row.list_id << '\n';
    }
}

std::vector<SubmissionRow> load_submission_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open submission CSV: " + path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Submission CSV is empty");
    }

    const std::vector<std::string> header = parse_csv_line(line);
    const std::size_t wave_column = require_column(header, "WAVE_ID");
    const std::size_t order_column = require_column(header, "ORDER_ID");
    const std::size_t list_column = require_column(header, "LIST_ID");

    std::vector<SubmissionRow> rows;
    while (std::getline(input, line)) {
        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.empty() || (fields.size() == 1U && fields[0].empty())) {
            continue;
        }
        if (fields.size() < header.size()) {
            fields.resize(header.size());
        }
        rows.push_back({fields[wave_column], fields[order_column], fields[list_column]});
    }

    return rows;
}

std::vector<std::string> validate_submission_rows(const ProblemInstance& instance, const std::vector<SubmissionRow>& rows) {
    std::vector<std::string> errors;
    std::unordered_map<std::string, int> assignments_by_order;
    std::unordered_map<std::string, std::string> wave_by_list;
    std::unordered_map<std::string, std::set<std::string>> orders_by_list;
    std::set<std::tuple<std::string, std::string, std::string>> unique_rows;
    std::set<std::pair<std::string, std::string>> unique_wave_orders;

    for (const SubmissionRow& row : rows) {
        if (row.wave_id.empty() || row.order_id.empty() || row.list_id.empty()) {
            errors.push_back("Submission contains an empty WAVE_ID, ORDER_ID, or LIST_ID");
            continue;
        }

        const auto order_it = instance.order_index_by_id.find(row.order_id);
        if (order_it == instance.order_index_by_id.end()) {
            errors.push_back("Submission contains unknown ORDER_ID: " + row.order_id);
            continue;
        }

        if (!instance.wave_index_by_id.contains(row.wave_id)) {
            errors.push_back("Submission contains unknown WAVE_ID: " + row.wave_id);
            continue;
        }

        const Order& order = instance.orders.at(static_cast<std::size_t>(order_it->second));
        if (order.wave_id != row.wave_id) {
            errors.push_back("Order " + row.order_id + " is assigned to the wrong wave");
        }

        assignments_by_order[row.order_id] += 1;
        unique_wave_orders.emplace(row.wave_id, row.order_id);
        if (!unique_rows.emplace(row.wave_id, row.order_id, row.list_id).second) {
            errors.push_back("Submission contains a duplicate row for ORDER_ID " + row.order_id);
        }

        const auto list_it = wave_by_list.find(row.list_id);
        if (list_it == wave_by_list.end()) {
            wave_by_list.emplace(row.list_id, row.wave_id);
        } else if (list_it->second != row.wave_id) {
            errors.push_back("LIST_ID " + row.list_id + " mixes orders from multiple waves");
        }
        orders_by_list[row.list_id].insert(row.order_id);
    }

    for (const Order& order : instance.orders) {
        const auto assignment_it = assignments_by_order.find(order.id);
        const int assignment_count = assignment_it == assignments_by_order.end() ? 0 : assignment_it->second;
        if (assignment_count != 1) {
            errors.push_back("ORDER_ID " + order.id + " appears " + std::to_string(assignment_count) + " times");
        }
    }

    for (const auto& [list_id, orders] : orders_by_list) {
        if (orders.size() > instance.nmax) {
            errors.push_back("LIST_ID " + list_id + " contains more than " + std::to_string(instance.nmax) + " orders");
        }
    }

    if (unique_wave_orders.size() != instance.orders.size()) {
        errors.push_back("Submission does not contain the expected number of unique (WAVE_ID, ORDER_ID) pairs");
    }

    return errors;
}

double score_submission_rows(const ProblemInstance& instance, const std::vector<SubmissionRow>& rows) {
    const std::vector<std::string> errors = validate_submission_rows(instance, rows);
    if (!errors.empty()) {
        throw std::runtime_error("Cannot score an invalid submission");
    }

    std::unordered_map<std::string, std::vector<int>> orders_by_list;
    std::unordered_map<std::string, std::string> wave_by_list;
    for (const SubmissionRow& row : rows) {
        orders_by_list[row.list_id].push_back(instance.order_index_by_id.at(row.order_id));
        wave_by_list[row.list_id] = row.wave_id;
    }

    BatchEvaluator evaluator(instance);
    double objective = 0.0;
    for (const auto& [list_id, order_indices] : orders_by_list) {
        (void)list_id;
        const int wave_index = instance.wave_index_by_id.at(wave_by_list.at(list_id));
        objective += evaluator.batch_cost(wave_index, order_indices);
    }
    return objective;
}

}  // namespace optioryx
