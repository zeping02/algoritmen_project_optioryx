#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

namespace optioryx {

using ConfigMap = std::unordered_map<std::string, std::string>;

ConfigMap load_config(const std::filesystem::path& path);

std::string config_string(const ConfigMap& config, const std::string& key, const std::string& fallback);
int config_int(const ConfigMap& config, const std::string& key, int fallback);
std::size_t config_size(const ConfigMap& config, const std::string& key, std::size_t fallback);
double config_double(const ConfigMap& config, const std::string& key, double fallback);
bool config_bool(const ConfigMap& config, const std::string& key, bool fallback);

}  // namespace optioryx
