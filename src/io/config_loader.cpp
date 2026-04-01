#include "optioryx/io/config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

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

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

}  // namespace

ConfigMap load_config(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open config file: " + path.string());
    }

    ConfigMap config;
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

        const std::size_t separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Invalid config line: " + line);
        }

        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1U));
        config[key] = value;
    }

    return config;
}

std::string config_string(const ConfigMap& config, const std::string& key, const std::string& fallback) {
    const auto it = config.find(key);
    return it == config.end() ? fallback : it->second;
}

int config_int(const ConfigMap& config, const std::string& key, int fallback) {
    const auto it = config.find(key);
    return it == config.end() ? fallback : std::stoi(it->second);
}

std::size_t config_size(const ConfigMap& config, const std::string& key, std::size_t fallback) {
    const auto it = config.find(key);
    return it == config.end() ? fallback : static_cast<std::size_t>(std::stoll(it->second));
}

double config_double(const ConfigMap& config, const std::string& key, double fallback) {
    const auto it = config.find(key);
    return it == config.end() ? fallback : std::stod(it->second);
}

bool config_bool(const ConfigMap& config, const std::string& key, bool fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }

    const std::string value = lowercase(it->second);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Invalid boolean config value for " + key + ": " + it->second);
}

}  // namespace optioryx
