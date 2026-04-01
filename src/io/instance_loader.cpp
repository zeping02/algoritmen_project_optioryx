#include "optioryx/io/instance_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace optioryx {

namespace {

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

}  // namespace

ProblemInstance load_instance(const std::filesystem::path& path) {
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
            instance.depot_index = depot_it->second;
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

}  // namespace optioryx
