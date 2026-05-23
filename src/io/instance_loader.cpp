#include "optioryx/io/instance_loader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "optioryx/io/csv_utils.hpp"

namespace optioryx {

namespace {

struct DistanceMatrixData {
    std::vector<std::string> item_ids;
    std::vector<std::vector<double>> distances;
    std::unordered_map<std::string, int> item_index_by_id;
};

std::string strip_utf8_bom(std::string value) {
    constexpr unsigned char bom0 = 0xEF;
    constexpr unsigned char bom1 = 0xBB;
    constexpr unsigned char bom2 = 0xBF;
    if (value.size() >= 3U &&
        static_cast<unsigned char>(value[0]) == bom0 &&
        static_cast<unsigned char>(value[1]) == bom1 &&
        static_cast<unsigned char>(value[2]) == bom2) {
        return value.substr(3);
    }
    return value;
}

std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

std::vector<std::string> split_tokens(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::size_t require_column(const std::vector<std::string>& header, const std::string& column_name) {
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (trim(header[index]) == column_name) {
            return index;
        }
    }
    throw std::runtime_error("Missing required column: " + column_name);
}

std::vector<int> sort_and_deduplicate(const std::vector<int>& values) {
    std::vector<int> result = values;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

DistanceMatrixData read_distance_matrix(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open distance matrix: " + path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Distance matrix is empty: " + path.string());
    }

    std::vector<std::string> header = parse_csv_line(strip_utf8_bom(line));
    if (header.size() < 2U) {
        throw std::runtime_error("Distance matrix header is invalid");
    }

    DistanceMatrixData data;
    data.item_ids.assign(header.begin() + 1, header.end());
    for (std::size_t index = 0; index < data.item_ids.size(); ++index) {
        data.item_ids[index] = trim(data.item_ids[index]);
        if (data.item_ids[index].empty()) {
            throw std::runtime_error("Distance matrix contains an empty location name");
        }
        data.item_index_by_id.emplace(data.item_ids[index], static_cast<int>(index));
    }

    std::unordered_map<std::string, std::vector<double>> row_values_by_name;
    while (std::getline(input, line)) {
        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.empty() || trim(fields[0]).empty()) {
            continue;
        }
        if (fields.size() != data.item_ids.size() + 1U) {
            throw std::runtime_error("Distance matrix row has the wrong width");
        }

        const std::string row_name = trim(fields[0]);
        std::vector<double> values;
        values.reserve(data.item_ids.size());
        for (std::size_t column = 1; column < fields.size(); ++column) {
            values.push_back(std::stod(trim(fields[column])));
        }
        row_values_by_name[row_name] = values;
    }

    data.distances.resize(data.item_ids.size(), std::vector<double>(data.item_ids.size(), 0.0));
    for (std::size_t row_index = 0; row_index < data.item_ids.size(); ++row_index) {
        const auto row_it = row_values_by_name.find(data.item_ids[row_index]);
        if (row_it == row_values_by_name.end()) {
            throw std::runtime_error("Distance matrix is missing row for location: " + data.item_ids[row_index]);
        }
        data.distances[row_index] = row_it->second;
    }

    return data;
}

void finalize_snake_rank(
    ProblemInstance& instance,
    const std::unordered_map<std::string, int>& asis_sequence_by_location) {
    instance.snake_rank.assign(instance.item_ids.size(), static_cast<int>(instance.item_ids.size() + 1U));

    int next_rank = 1;
    std::vector<std::pair<int, int>> ranked_locations;
    ranked_locations.reserve(asis_sequence_by_location.size());
    for (const auto& [location_id, sequence] : asis_sequence_by_location) {
        const auto item_it = instance.item_index_by_id.find(location_id);
        if (item_it == instance.item_index_by_id.end()) {
            throw std::runtime_error("Unknown location in ASIS data: " + location_id);
        }
        ranked_locations.emplace_back(sequence, item_it->second);
    }

    std::sort(ranked_locations.begin(), ranked_locations.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) {
            return left.first < right.first;
        }
        return left.second < right.second;
    });

    instance.snake_rank[static_cast<std::size_t>(instance.start_index)] = 0;
    for (const auto& [sequence, item_index] : ranked_locations) {
        (void)sequence;
        if (item_index == instance.start_index || item_index == instance.end_index) {
            continue;
        }
        instance.snake_rank[static_cast<std::size_t>(item_index)] = next_rank;
        ++next_rank;
    }

    if (instance.end_index != instance.start_index) {
        instance.snake_rank[static_cast<std::size_t>(instance.end_index)] = next_rank;
        ++next_rank;
    }

    for (std::size_t item_index = 0; item_index < instance.snake_rank.size(); ++item_index) {
        if (instance.snake_rank[item_index] <= static_cast<int>(instance.item_ids.size())) {
            continue;
        }
        instance.snake_rank[item_index] = next_rank;
        ++next_rank;
    }
}

std::filesystem::path resolve_competition_root(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path;
    }
    if (path.filename() == "pick_data.csv") {
        return path.parent_path();
    }
    if (path.extension() == ".csv" && std::filesystem::exists(path.parent_path() / "pick_data.csv")) {
        return path.parent_path();
    }
    return {};
}

}  // namespace

ProblemInstance load_fixture_instance(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open instance file: " + path.string());
    }

    ProblemInstance instance;
    std::vector<std::string> snake_items;
    std::vector<std::vector<double>> distances;
    std::vector<Order> orders;
    std::unordered_map<std::string, int> wave_index_by_id;

    enum class Section { root, distances, orders };
    Section section = Section::root;

    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comment_start = line.find('#');
        if (comment_start != std::string::npos) {
            line = line.substr(0, comment_start);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        if (line == "DISTANCES") {
            section = Section::distances;
            continue;
        }
        if (line == "END_DISTANCES") {
            section = Section::root;
            continue;
        }
        if (line == "ORDERS") {
            section = Section::orders;
            continue;
        }
        if (line == "END_ORDERS") {
            section = Section::root;
            continue;
        }

        if (section == Section::distances) {
            const std::vector<std::string> tokens = split_tokens(line);
            std::vector<double> row;
            row.reserve(tokens.size());
            for (const std::string& token : tokens) {
                row.push_back(std::stod(token));
            }
            distances.push_back(row);
            continue;
        }

        if (section == Section::orders) {
            const std::vector<std::string> tokens = split_tokens(line);
            if (tokens.size() < 3U) {
                throw std::runtime_error("Invalid order line: " + line);
            }

            Order order;
            order.id = tokens[0];
            order.wave_id = tokens[1];
            if (instance.order_index_by_id.contains(order.id)) {
                throw std::runtime_error("Duplicate order id in fixture instance: " + order.id);
            }

            for (std::size_t index = 2; index < tokens.size(); ++index) {
                const auto item_it = instance.item_index_by_id.find(tokens[index]);
                if (item_it == instance.item_index_by_id.end()) {
                    throw std::runtime_error("Unknown item in order: " + tokens[index]);
                }
                order.items.push_back(item_it->second);
            }

            if (!wave_index_by_id.contains(order.wave_id)) {
                Wave wave;
                wave.id = order.wave_id;
                instance.waves.push_back(wave);
                wave_index_by_id.emplace(order.wave_id, static_cast<int>(instance.waves.size() - 1U));
            }

            const int order_index = static_cast<int>(orders.size());
            instance.order_index_by_id.emplace(order.id, order_index);
            orders.push_back(order);
            instance.waves.at(static_cast<std::size_t>(wave_index_by_id.at(order.wave_id))).order_indices.push_back(order_index);
            continue;
        }

        if (starts_with(line, "NAME ")) {
            instance.name = trim(line.substr(5));
            continue;
        }
        if (starts_with(line, "NMAX ")) {
            instance.nmax = static_cast<std::size_t>(std::stoll(trim(line.substr(5))));
            continue;
        }
        if (starts_with(line, "ITEMS ")) {
            const std::vector<std::string> item_ids = split_tokens(line.substr(6));
            instance.item_ids = item_ids;
            instance.item_index_by_id.clear();
            for (std::size_t index = 0; index < item_ids.size(); ++index) {
                instance.item_index_by_id.emplace(item_ids[index], static_cast<int>(index));
            }
            continue;
        }
        if (starts_with(line, "DEPOT ")) {
            const std::string depot_id = trim(line.substr(6));
            const auto depot_it = instance.item_index_by_id.find(depot_id);
            if (depot_it == instance.item_index_by_id.end()) {
                throw std::runtime_error("DEPOT must refer to an item declared in ITEMS");
            }
            instance.start_index = depot_it->second;
            instance.end_index = depot_it->second;
            continue;
        }
        if (starts_with(line, "START ")) {
            const std::string start_id = trim(line.substr(6));
            const auto start_it = instance.item_index_by_id.find(start_id);
            if (start_it == instance.item_index_by_id.end()) {
                throw std::runtime_error("START must refer to an item declared in ITEMS");
            }
            instance.start_index = start_it->second;
            continue;
        }
        if (starts_with(line, "END ")) {
            const std::string end_id = trim(line.substr(4));
            const auto end_it = instance.item_index_by_id.find(end_id);
            if (end_it == instance.item_index_by_id.end()) {
                throw std::runtime_error("END must refer to an item declared in ITEMS");
            }
            instance.end_index = end_it->second;
            continue;
        }
        if (starts_with(line, "SNAKE ")) {
            snake_items = split_tokens(line.substr(6));
            continue;
        }

        throw std::runtime_error("Unknown line in instance file: " + line);
    }

    if (instance.item_ids.empty()) {
        throw std::runtime_error("Instance is missing ITEMS");
    }
    if (instance.nmax == 0U) {
        throw std::runtime_error("Instance must define NMAX > 0");
    }
    if (distances.size() != instance.item_ids.size()) {
        throw std::runtime_error("Distance row count does not match item count");
    }
    for (const std::vector<double>& row : distances) {
        if (row.size() != instance.item_ids.size()) {
            throw std::runtime_error("Distance matrix must be square");
        }
    }

    if (snake_items.empty()) {
        snake_items = instance.item_ids;
    }
    if (snake_items.size() != instance.item_ids.size()) {
        throw std::runtime_error("SNAKE item count must match ITEMS");
    }

    instance.snake_rank.assign(instance.item_ids.size(), -1);
    for (std::size_t index = 0; index < snake_items.size(); ++index) {
        const auto item_it = instance.item_index_by_id.find(snake_items[index]);
        if (item_it == instance.item_index_by_id.end()) {
            throw std::runtime_error("Unknown item in SNAKE section: " + snake_items[index]);
        }
        instance.snake_rank.at(static_cast<std::size_t>(item_it->second)) = static_cast<int>(index);
    }
    for (const int rank : instance.snake_rank) {
        if (rank < 0) {
            throw std::runtime_error("SNAKE section must list every item exactly once");
        }
    }

    instance.distances = std::move(distances);
    instance.orders = std::move(orders);
    instance.wave_index_by_id = std::move(wave_index_by_id);
    return instance;
}

ProblemInstance load_competition_instance(const std::filesystem::path& path) {
    const std::filesystem::path root = resolve_competition_root(path);
    if (root.empty()) {
        throw std::runtime_error("Competition input must be a directory containing pick_data.csv and dist_matrix.csv");
    }

    const DistanceMatrixData matrix = read_distance_matrix(root / "dist_matrix.csv");

    ProblemInstance instance;
    instance.name = root.filename().empty() ? "competition_data" : root.filename().string();
    instance.nmax = 20;
    instance.item_ids = matrix.item_ids;
    instance.item_index_by_id = matrix.item_index_by_id;
    instance.distances = matrix.distances;

    const auto start_it = instance.item_index_by_id.find("START");
    const auto end_it = instance.item_index_by_id.find("END");
    if (start_it == instance.item_index_by_id.end() || end_it == instance.item_index_by_id.end()) {
        throw std::runtime_error("Competition distance matrix must contain START and END");
    }
    instance.start_index = start_it->second;
    instance.end_index = end_it->second;

    std::ifstream input(root / "pick_data.csv");
    if (!input) {
        throw std::runtime_error("Unable to open pick_data.csv");
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("pick_data.csv is empty");
    }

    std::vector<std::string> header = parse_csv_line(strip_utf8_bom(line));
    const std::size_t wave_column = require_column(header, "WAVE_ID");
    const std::size_t order_column = require_column(header, "ORDER_ID");
    const std::size_t location_column = require_column(header, "LOCATION_ID");
    const std::size_t sequence_column = require_column(header, "ASIS_SEQUENCE");

    std::unordered_map<std::string, int> asis_sequence_by_location;
    while (std::getline(input, line)) {
        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.size() < header.size()) {
            fields.resize(header.size());
        }

        const std::string wave_id = trim(fields[wave_column]);
        const std::string order_id = trim(fields[order_column]);
        const std::string location_id = trim(fields[location_column]);
        const int asis_sequence = std::stoi(trim(fields[sequence_column]));

        if (wave_id.empty() || order_id.empty() || location_id.empty()) {
            throw std::runtime_error("pick_data.csv contains an empty WAVE_ID, ORDER_ID, or LOCATION_ID");
        }

        const auto location_it = instance.item_index_by_id.find(location_id);
        if (location_it == instance.item_index_by_id.end()) {
            throw std::runtime_error("pick_data.csv references unknown LOCATION_ID: " + location_id);
        }

        const auto sequence_it = asis_sequence_by_location.find(location_id);
        if (sequence_it == asis_sequence_by_location.end()) {
            asis_sequence_by_location.emplace(location_id, asis_sequence);
        } else if (sequence_it->second != asis_sequence) {
            throw std::runtime_error("ASIS_SEQUENCE is not stable for location: " + location_id);
        }

        int order_index = -1;
        const auto order_it = instance.order_index_by_id.find(order_id);
        if (order_it == instance.order_index_by_id.end()) {
            Order order;
            order.id = order_id;
            order.wave_id = wave_id;
            order_index = static_cast<int>(instance.orders.size());
            instance.orders.push_back(order);
            instance.order_index_by_id.emplace(order_id, order_index);

            if (!instance.wave_index_by_id.contains(wave_id)) {
                Wave wave;
                wave.id = wave_id;
                instance.waves.push_back(wave);
                instance.wave_index_by_id.emplace(wave_id, static_cast<int>(instance.waves.size() - 1U));
            }
            instance.waves.at(static_cast<std::size_t>(instance.wave_index_by_id.at(wave_id))).order_indices.push_back(order_index);
        } else {
            order_index = order_it->second;
            if (instance.orders.at(static_cast<std::size_t>(order_index)).wave_id != wave_id) {
                throw std::runtime_error("ORDER_ID appears in multiple waves: " + order_id);
            }
        }

        instance.orders.at(static_cast<std::size_t>(order_index)).items.push_back(location_it->second);
    }

    for (Order& order : instance.orders) {
        order.items = sort_and_deduplicate(order.items);
    }

    finalize_snake_rank(instance, asis_sequence_by_location);
    return instance;
}

ProblemInstance load_instance(const std::filesystem::path& path) {
    if (const std::filesystem::path root = resolve_competition_root(path); !root.empty()) {
        return load_competition_instance(root);
    }
    return load_fixture_instance(path);
}

}  // namespace optioryx
